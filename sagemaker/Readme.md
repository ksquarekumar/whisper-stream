# Building custom container images for SageMaker

- references:
  - [sagemaker-studio/deep-learning-containers-images](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/deep-learning-containers-images.html)
  - [sagemaker-studio-bring-your-own-images](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-byoi.html)
  - [sagemaker-studio/custom-image-samples](https://github.com/aws-samples/sagemaker-studio-custom-image-samples/blob/main/examples/conda-env-kernel-image/README.md)
  - [jax_bring_your_own/docker](https://github.com/aws/amazon-sagemaker-examples/tree/main/advanced_functionality/jax_bring_your_own/docker)
  - [conda-forge/customizing-cuda](https://conda-forge.org/docs/user/tipsandtricks.html#installing-cuda-enabled-packages-like-tensorflow-and-pytorch)
  - [jax/installation](https://github.com/google/jax#installation)
  - [sagemaker-studio/image-build-cli](https://github.com/aws-samples/sagemaker-studio-image-build-cli)
  - [sagemaker/deep-learning-containers/available_images](https://github.com/aws/deep-learning-containers/blob/master/available_images.md)
  - [sagemaker-studio/custom-image-example](https://github.com/aws-samples/sagemaker-studio-custom-image-samples/tree/main/examples/conda-env-kernel-image)
  - [sagemaker/source-container](https://github.com/aws/deep-learning-containers/blob/master/pytorch/inference/docker/2.0/py3/cu118/Dockerfile.gpu)

## Set environment variables

```shell
USERNAME="<your_username>"
EMAIL_ADDRESS="<your_email>"
SOURCE_REGION="${AWS_REGION}"
SOURCE_ACCOUNT_ID="763104351884"
SOURCE_IMAGE_NAME="pytorch-training"
SOURCE_ECR=${SOURCE_IMAGE_NAME}
SOURCE_IMAGE_VERSION="2.0.1-gpu-py310-cu118-ubuntu20.04-ec2"
DESTINATION_ACCOUNT_ID="<Your_account_ID>"
DESTINATION_REGION="${AWS_REGION}"
DESTINATION_REPO_NAME="smstudio-custom"
DESTINATION_ECR="${DESTINATION_REPO_NAME}"
DESTINATION_IMAGE_VERSION="jax-cu11"
ROLE_ARN="<YOUR_ROLE_ARN>"
DOMAIN_ID="<YOUR_SAGEMAKER_DOMAIN_ID>"
```

## Authenticate with the source `ECR`

```shell
aws --region ${SOURCE_REGION} ecr get-login-password | docker login --username AWS --password-stdin "${SOURCE_ACCOUNT_ID}.dkr.ecr.${SOURCE_REGION}.amazonaws.com/${SOURCE_ECR}"
```

## Create Destination `ECR`

```shell
aws ecr create-repository --repository-name "${DESTINATION_ECR}"
```

## Authenticate with destination ecr

```shell
aws --region ${DESTINATION_REGION} ecr get-login-password | docker login --username AWS --password-stdin "${DESTINATION_ACCOUNT_ID}.dkr.ecr.${DESTINATION_REGION}.amazonaws.com/${DESTINATION_ECR}"
```

## Build image locally

```shell
# Build and push the image
docker buildx build \
  --progress=plain \
  -f sagemaker/Dockerfile.sagemaker \
  --cache-from "${DESTINATION_ACCOUNT_ID}.dkr.ecr.${DESTINATION_REGION}.amazonaws.com/${DESTINATION_REPO_NAME}:${DESTINATION_IMAGE_VERSION}" \
  --build-arg USER_NAME="${USER_NAME}" \
  --build-arg EMAIL_ADDRESS="${EMAIL_ADDRESS}" \
  --build-arg SOURCE_ACCOUNT_ID="${SOURCE_ACCOUNT_ID}" \
  --build-arg SOURCE_REGION="${SOURCE_REGION}" \
  --build-arg SOURCE_IMAGE_NAME="${SOURCE_IMAGE_NAME}" \
  --build-arg SOURCE_IMAGE_VERSION="${SOURCE_IMAGE_VERSION}" \
  -t "${DESTINATION_IMAGE_VERSION}" \
  -t "${DESTINATION_ACCOUNT_ID}.dkr.ecr.${DESTINATION_REGION}.amazonaws.com/${DESTINATION_REPO_NAME}:${DESTINATION_IMAGE_VERSION}" \
  "./sagemaker"
```

## Testing the built image

### Verify the Image runs

```shell
docker run -it "${DESTINATION_IMAGE_VERSION}" bash
```

### Verify the Image is `KernelApp` Compatible, i.e. the kernels in the image can be made visible to a `Jupyter Kernel Gateway`.

#### Run the container with a `KernelGateway` to validate that the kernels are visible from the `REST` endpoint exposed on the local host.

```shell
docker run -it -p 8888:8888 "${DESTINATION_IMAGE_VERSION}" bash -c 'pip install jupyter_kernel_gateway && jupyter kernelgateway --ip 0.0.0.0 --debug --port 8888'

curl http://0.0.0.0:8888/api/kernelspecs
```

## push image to `ECR`

```shell
docker push "${DESTINATION_ACCOUNT_ID}.dkr.ecr.${DESTINATION_REGION}.amazonaws.com/${DESTINATION_REPO_NAME}:${DESTINATION_IMAGE_VERSION}"
```

## Using this `image` with sageMaker studio

### Create a SageMaker Image (SMI) `image` and an `image-version` with the newly built image in `ECR`.

- After creating an `SMI` ` Image``, create an  `Image-Version` during which SageMaker stores image metadata like ECR / SHA / BACKEND TYPES etc.
- Request parameter `RoleArn` value is used to get ECR image information when an `Image-Version` is created.
- _A **new** `Image-Version` should be created every time an image is updated in ECR._

```shell
# Create Parent SMI Image (Only once per ECR Image)
aws --region "${DESTINATION_REGION}" sagemaker create-image \
  --description "Custom container image for SageMaker with JAX" \
  --image-name "${DESTINATION_IMAGE_VERSION}" \
  --display-name "${DESTINATION_IMAGE_VERSION}" \
  --role-arn "${ROLE_ARN}"

# Create an Image-Version for the Image (For every image rebuild)
aws --region "${DESTINATION_REGION}" sagemaker create-image-version \
  --image-name "${DESTINATION_IMAGE_VERSION}" \
  --aliases jax cu11 \
  --base-image "${DESTINATION_ACCOUNT_ID}.dkr.ecr.${DESTINATION_REGION}.amazonaws.com/${DESTINATION_REPO_NAME}:${DESTINATION_IMAGE_VERSION}" \
  --job-type "NOTEBOOK_KERNEL" \
  --ml-framework "jax 4.14" \
  --programming-lang "python 3.10" \
  --processor "GPU" \
  --horovod \
  --release-notes "Support for Jax >=4.14"
```

### Verify the `image-version` is created successfully.

- **Do NOT proceed** if `Image-Version` is in `CREATE_FAILED` state or in any other state apart from `CREATED`.

```shell
aws --region "${DESTINATION_REGION}" sagemaker describe-image-version --image-name "${DESTINATION_IMAGE_VERSION}"
```

### Create an `AppImageConfig` for this image

- This is the underlying config that defines the run time behavior, and is a general config that can be attached to any image that works with it.
- Need to do it only once per defined `ECR/SMI` image, and although it can be shared across multiple `SMI` images, it is recommended to keep it `1:1` for `SMI:Environment` better reproducibility.

```shell
aws --region "${DESTINATION_REGION}" sagemaker create-app-image-config \
  --app-image-config-name "${DESTINATION_IMAGE_VERSION}-config" \
  --kernel-gateway-image-config \
  '{
	"KernelSpecs": [{
		"Name": "python3",
		"DisplayName": "Python [conda env: base]"
	    }],
	"FileSystemConfig": {
		"MountPath": "/root",
		"DefaultUid": 0,
		"DefaultGid": 0
    }'
```

### Update your `user` and `domain-id` to use the newly created `AppImageConfig`

- This is where the `AppConfig` gets attached to our `SMI` Image.

```shell
aws --region "${DESTINATION_REGION}" sagemaker update-domain \
  --domain-id "${DOMAIN_ID}" \
  --default-user-settings \
  '{
    "KernelGatewayAppSettings": {
    "LifecycleConfigArns": [],
    "CustomImages": [
                {
                    "ImageName": "${DESTINATION_IMAGE_VERSION}",
                    "AppImageConfigName": "jax-cu11-config"
                }
            ]
    }'
```

```shell
aws --region "${DESTINATION_REGION}" sagemaker update-user-profile \
  --domain-id "${DOMAIN_ID}" \
  --user-profile-name "${USERNAME}"
--user-settings \
  '{
    "KernelGatewayAppSettings": {
    "LifecycleConfigArns": [],
    "CustomImages": [
                {
                    "ImageName": "${DESTINATION_IMAGE_VERSION}",
                    "AppImageConfigName": "jax-cu11-config"
                }
            ]
    }'
```

---
