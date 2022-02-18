from typing import Callable, List, Optional

import os
import time
import json
from pathlib import Path
from io import BytesIO
from PIL import Image
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.exceptions import ClientError
import sagemaker
from sagemaker import image_uris
from sagemaker.pytorch import PyTorch, PyTorchModel
from sagemaker.serializers import NumpySerializer, JSONSerializer
from sagemaker.deserializers import NumpyDeserializer, JSONDeserializer

import typer
from concurrent.futures import ThreadPoolExecutor
from decord import VideoReader

from mlserve.common.logger import logger

app = typer.Typer(name="Sagemaker real time inference", add_completion=False)

def model_exists(model_name: str) -> bool:
    sm_client = boto3.client('sagemaker')
    
    models_present: List = []

    for model_info in sm_client.list_models(NameContains=model_name)['Models']:
        logger.info(f"Checking model: {model_info['ModelName']}")
        if model_info['ModelName'] == model_name:
            logger.info(f"Model found: {model_info}")
            return True
        else:
            models_present.append(model_info['ModelName'])
    
    logger.info(f"The models present in Sagemaker are: {models_present}")

    return False


def endpoint_configs_exists(config_name:str) -> bool:
    sm_client = boto3.client('sagemaker')
    
    configs_present: List = []

    for config_info in sm_client.list_endpoint_configs(NameContains=config_name)['EndpointConfigs']:
        logger.info(f"Checking model: {config_info['EndpointConfigName']}")
        if config_info['EndpointConfigName'] == config_name:
            logger.info(f"Config found: {config_info}")
            return True
        else:
            configs_present.append(config_info['EndpointConfigName'])
    
    logger.info(f"The models present in Sagemaker are: {configs_present}")

    return False

class FVecsRealTime():

    def __init__(
        self,
        model_name:str,
        model_s3_path:str,
        local_model:bool=True,
        instance_type:Optional[str]=None,
        instance_count:Optional[int]=1 ,
        image_uri:Optional[str]=None,
        framework_version:str="1.10.0",
        py_version:str="py38",
    ) -> None:
        super().__init__()

        self.local_mode = local_model
        self.py_version = py_version
        self.framework_version = framework_version
        
        self.region = boto3.Session().region_name
        self.sm_client = boto3.client("sagemaker", region_name=self.region)
        
        try:
            self.role = sagemaker.get_execution_role()
        except ValueError:
            iam = boto3.client('iam')
            self.role = iam.get_role(RoleName=os.environ['AWS_ROLE_NAME'])['Role']['Arn']
            
        logger.info(f"Obtained role: {self.role}")
            
        #Get model from S3
        self.model_uri = model_s3_path
        logger.info(f"Loading model from {self.model_uri}")

        if self.local_mode:
            self.instance_type = 'local'
        else:
            self.instance_count = instance_count
            if instance_type is None:
                self.instance_type = 'ml.m4.xlarge'
            else:
                self.instance_type = instance_type

        logger.info(f"Setting instance type to: {self.instance_type}")
        #Get container image (prebuilt example: )
        if image_uri is None:
            try:
                self.image_uri = image_uris.retrieve(
                    framework='pytorch',
                    region=self.region,
                    image_scope="inference",
                    py_version=py_version,
                    version=framework_version,
                    instance_type='ml.m4.xlarge'
                )
            except Exception as e:
                #self.image_uri = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38"
                self.image_uri = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.10.0-gpu-py38"
        else:
            self.image_uri = image_uri

        logger.info(f"Setting image uri: {self.image_uri}")

        self.model_name = model_name
        self.endpoint_config_name = f"{self.model_name}-config"
        self.endpoint_name = f"{self.model_name}-endpt"
        
        self.smr = None
        
    def create_model(self):
        reference_container = {
            "Image": self.image_uri,
            "ModelDataUrl": self.model_uri
        }
        
        create_model_response = self.sm_client.create_model(
            ModelName = self.model_name,
            ExecutionRoleArn = self.role,
            PrimaryContainer= reference_container)

        logger.info(f"Model Arn: {create_model_response['ModelArn']}")

    def create_endpoint_config(self):
        create_endpoint_config_response = self.sm_client.create_endpoint_config(
            EndpointConfigName = self.endpoint_config_name,
            ProductionVariants=[{
                'InstanceType': self.instance_type,
                'InitialInstanceCount': 1,
                'InitialVariantWeight': 1,
                'ModelName': self.model_name,
                'VariantName': 'AllTraffic',
                }])

        logger.info(f"Endpoint config Arn: {create_endpoint_config_response['EndpointConfigArn']}")

    def create_endpoint(self):
        
        create_endpoint_response = self.sm_client.create_endpoint(
            EndpointName=self.endpoint_name,
            EndpointConfigName=self.endpoint_config_name)
        logger.info('Endpoint Arn: ' + create_endpoint_response['EndpointArn'])
        
        resp = self.sm_client.describe_endpoint(EndpointName=self.endpoint_name)
        status = resp['EndpointStatus']
        logger.info("Endpoint Status: " + status)
        
        logger.info('Waiting for {} endpoint to be in service...'.format(self.endpoint_name))
        waiter = self.sm_client.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=self.endpoint_name)

        logger.info(f"Model ({self.model_name} real time deployed)")

    def launch_server(self) -> None:
        
        # Launching the model
        logger.info(f"Creating a new model from s3: {self.model_uri}")
        self.create_model()

        logger.info(f"Creating Endpoint config name: {self.endpoint_config_name}")
        self.create_endpoint_config()

        logger.info(f'Setting up end point: {self.endpoint_name}')
        self.create_endpoint()

    def deploy_endpoint(self,
        batch_size:int,
        num_min_workers:int,
        num_max_workers:int,
        accelerate:bool=False
    ) -> None:
        env_variables_dict = {
            "SAGEMAKER_TS_MIN_WORKERS": num_min_workers.__str__(),
            "SAGEMAKER_TS_MAX_WORKERS": num_max_workers.__str__(),
            "SAGEMAKER_TS_BATCH_SIZE": batch_size.__str__(),
            "SAGEMAKER_TS_MAX_BATCH_DELAY": "100000"
        }
        
        logger.info(f"Deployment config: {env_variables_dict}")

        self.pytorch_model = PyTorchModel(
            model_data=self.model_uri,
            role=self.role,
            image_uri=self.image_uri,
            #source_dir="code",
            framework_version=self.framework_version,
            entry_point="inference.py",
            env=env_variables_dict
        )
        
        logger.info(f"Deploying the model at {self.instance_type}")
        
        deploy_args = {
            "initial_instance_count":1,
            "instance_type":self.instance_type,
            "serializer":NumpySerializer(),
            "deserializer":NumpyDeserializer()
        }
        
        if accelerate:
            deploy_args.update({

            })

        self.predictor = self.pytorch_model.deploy(**deploy_args)

    def deploym_predict(self, frms: np.array, mthreading:bool=False) -> np.ndarray:
        fvecs_list = []
        
        if mthreading:
            with ThreadPoolExecutor() as executor:
                fvecs_list:List[np.ndarray] = list(executor.map(lambda frm: self.predictor.predict(np.expand_dims(frm, axis=0).astype(np.float32)), frms))
        else:
            pbar = tqdm(enumerate(frms), total=frms.shape[0], desc="Pinging endpoint")
            for idx, frm in pbar:
                fvecs_list.append(self.predictor.predict(np.expand_dims(frm, axis=0).astype(np.float32)))

        self.predictor.delete_endpoint()

        return np.array(fvecs_list)

    def create_pred_client(self):
        logger.info("Creating Sagemaker runtime client")
        self.smr = boto3.client('sagemaker-runtime')

    def predict(self, frm:np.ndarray) -> np.ndarray:
        
        if self.smr is None:
            self.create_pred_client()

        if len(frm.shape) == 3:
            frm = np.expand_dims(frm, axis=0)

        srl_data = NumpySerializer().serialize(frm)

        try:
            resp = self.smr.invoke_endpoint(
                EndpointName=self.endpoint_name,
                Body=srl_data,
                ContentType='application/x-npy'
            )

            preds = np.array(json.loads(resp['Body'].read()))
    
        except Exception as e:
            logger.error("Issue with prediction from the server ...")
            preds = None

        return preds
    
    def get_predictions(self, frms: np.array):
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = executor.map(self.predict, frms)
            
        fvecs: np.ndarray = np.vstack(list(results))
        print(f"Results shape: {fvecs.shape}")
        
        return fvecs

    def prep_sagemaker(self):
        
        try:
            logger.info(f"Deleting endpoint: {self.endpoint_name}")
            self.sm_client.delete_endpoint(EndpointName=self.endpoint_name)
        except ClientError as error:
            if error.response['Error']['Code'] == "ValidationException":
                logger.warn(f"Could not find endpoint {self.endpoint_name}")
            else:
                raise error

        try:
            logger.info(f"Deleting end point config: {self.endpoint_config_name}")
            self.sm_client.delete_endpoint_config(EndpointConfigName=self.endpoint_config_name)
        except ClientError as error:
            if error.response['Error']['Code'] == "ValidationException":
                logger.warn(f"Could not find endpoint config {self.endpoint_config_name}")
            else:
                raise error

        try:
            logger.info(f"Deleting model: {self.model_name}")
            self.sm_client.delete_model(ModelName=self.model_name)
        except ClientError as error:
            if error.response['Error']['Code'] == "ValidationException":
                logger.warning(f"Could not find model : {self.model_name}")
            else:
                raise error

@app.command()
def main(
    model_name:Optional[str]=typer.Option(None, "-name", help="Model name"),
    vid_loc:Optional[Path]=typer.Option(None,"-i", help="Input Video file", exists=True),
    local_model:bool=typer.Option(False, "-local", help="Launching local server"),
    docker_image:Optional[str]=typer.Option(None, "-img", help="docker image name"),
    launch_fresh:bool=typer.Option(False, "-lf", help="clean up sagemaker"),
    launch_model:bool=typer.Option(False, "-lm", help="launch model on sagemaker"),
    get_preds:bool=typer.Option(False, "-gp", help="get predictions"),
    get_preds_batch:bool=typer.Option(False, "-gpb", help="get batch predictions"),
    check_pytorch_deploy:bool=typer.Option(False, "-cpd", help="Check pytorch deployment"),
    batch_size:int=typer.Option(1, "-bs", help="Batch size"),
    num_min_workers:int=typer.Option(1, "-minw", help="Num min workers"),
    num_max_workers:int=typer.Option(1, "-maxw", help="Num max workers"),
    gpu:bool=typer.Option(False, "-gpu",help="Use GPU"),
    thread_mode:bool=typer.Option(False, "--threads", help="Use threads in prediction"),
    accelerate:bool=typer.Option(False, "--accelerate", help="Use Elastic Inference")
):
    #mserver = FVecsRealTime(local_model=True, image_uri="pytorch-local:latest")
    
    if local_model and not check_pytorch_deploy:
        logger.error("local model only supported for pytorch deployment")
        raise typer.Exit()
    
    if model_name is None:
        model_name = 'fvecs-resnet18'

    server_config = {
        'model_name' : model_name,
        'model_s3_path' : 's3://amagitornado-test/Models/resnet18-fvecs/model.tar.gz',
        'local_model' : local_model
    }
    
    if gpu:
        docker_image = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.9.1-gpu-py38-cu111-ubuntu20.04"
    else:
        docker_image = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38"
    
    if docker_image is not None:
        server_config.update({'image_uri' : docker_image})

    mserver = FVecsRealTime(**server_config)
    
    if launch_fresh:
        logger.info(f"Deleting resources associated with {mserver.model_name}")
        mserver.prep_sagemaker()

    if launch_model:
        logger.info(f"Launching server for {mserver.model_name}")
        mserver.launch_server()
        
    if get_preds:
        num_data_pts = 2
        np_array = np.random.randint(256, size=num_data_pts*3*300*300).reshape(-1, 3, 300, 300) 
        logger.info("Server brought up ... Going to start the prediction next")
    
        preds = mserver.predict(np_array)

        logger.info(f"Returned np arr shape: {preds.shape}")
        
    if get_preds_batch:
        
        #vr = VideoReader('../../notebooks/scratch/BigBuckBunny_512kb.mp4')
        vr = VideoReader(vid_loc.resolve().__str__())
        np_array = vr.get_batch(range(1700)).asnumpy()
        #num_data_pts = 5
        #np_array = np.random.randint(256, size=num_data_pts*3*300*300).reshape(-1, 3, 300, 300) 
        logger.info("Server brought up ... Going to start the prediction next")
    
        start = time.perf_counter()
        preds = mserver.get_predictions(np_array)
        elapsed = time.perf_counter() - start
        logger.info(f"Time taken: {elapsed:0.2f} seconds")

        logger.info(f"Returned np arr shape: {preds.shape}")
        
    if check_pytorch_deploy:
        if vid_loc is None or not vid_loc.exists():
            logger.error("Need to give a valid video input path")
            raise typer.Exit()

        vr = VideoReader(vid_loc.resolve().__str__())
        np_array = vr.get_batch(range(1700)).asnumpy()
        
        start_serv_bringup = time.perf_counter()
        if num_max_workers < num_min_workers:
            num_max_workers = num_min_workers
        mserver.deploy_endpoint(batch_size, num_min_workers, num_max_workers)
        elapsed_time = time.perf_counter() - start_serv_bringup
        logger.info(f"Time taken for server bring up: {elapsed_time:.2f}")
        
        start_predict = time.perf_counter()
        #preds = mserver.deploym_predict(np_array)
        preds = mserver.deploym_predict(np_array, mthreading=thread_mode)
        elapsed_time = time.perf_counter() - start_predict
        logger.info(f"Got back preds ({preds.shape}) in {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    app()