from string import Template
from pathlib import Path
import subprocess, shlex
from typing import Optional

from mlserve.common.logger import logger

import typer

app = typer.Typer(name="Launch torchserve server", add_completion=False)

# Edit config.properties
def set_config_properties(
    batch_size:int,
    min_workers:int,
    max_workers:Optional[int]=None
    ) -> None:

    replacements = {
        "batch_size": str(batch_size),
        "min_workers": str(min_workers),
        'max_workers': str(max_workers or min_workers)
    }
    
    orig_txt = Path("template_config.properties").read_text()
    sub_txt = Template(orig_txt).substitute(replacements)
    Path('config.properties').write_text(sub_txt, encoding='utf-8')

#torch-model-archiver --model-name resnet18 \
#--version 1.0 \
#--serialized-file resnet18.pth \
#--extra-files ./FVHandler.py \
#--handler ./serve_handle.py  \
#--export-path model_store -f
@app.command()
def regen_mar(
    model_name:str=typer.Option('resnet18', "-mn", help="Model name"),
    model_info_dir:Path=typer.Option("model_info", "-mm", help="Model .pth file", exists=True),
    extra_files:str=typer.Option('FVHandler.py',"-ef", help="Extra Files"),
    handler:str=typer.Option('serve_handle.py', "-hd", help="Handler function"),
    model_store:str=typer.Option('model_store', "-ms", help="Model Store location"),
) -> str:
    
    model_pth = f"{model_info_dir}/{model_name}/{model_name}.pth"
    if not Path(model_pth).exists():
        typer.echo(f"{model_pth} does not exists")
        typer.Exit()
        
    # Currently support for a single file
    extra_files = f"{model_info_dir}/{model_name}/{extra_files}"
    handler = f"{model_info_dir}/{model_name}/{handler}"

    if not Path(model_store).exists():
        Path(model_store).mkdir(exist_ok=True)

    tmar_gen_cmd = f"torch-model-archiver --model-name {model_name} \
                    --version 1.0 \
                    --serialized-file {model_pth} \
                    --extra-files {extra_files} \
                    --handler {handler}  \
                    --export-path {model_store} -f"
    subprocess.run(shlex.split(tmar_gen_cmd))
    
    mar_file = Path(f"{model_store}/{model_name}.mar")
    if not mar_file.exists():
        typer.echo(f"Unable to create the {mar_file.resolve().__str__()} file")
        raise typer.Exit()
    else:
        typer.echo(f"Successfully created : {mar_file.resolve().__str__()}")
        
    return mar_file.resolve().__str__()

#torchserve --start --foreground --model-store model_store/ \
# --models resnet18=resnet18.mar --ts-config config.properties
@app.command()
def launch_local_server(
    model_name:str=typer.Option('resnet18', "-mn", help="Model name"),
    model_mar:str=typer.Option('resnet18.mar', "-mm", help="Model .mar file"),
    model_store:str=typer.Option('model_store', "-ms", help="Model Store location"),
    ts_config_file:str=typer.Option('config.properties', "-ts", help="TS config file")
) -> None:
    subprocess.run(shlex.split("rm -rf ./logs"))
    mar_file = Path(f"{model_store}/{model_mar}")

    launch_cmd = f"torchserve --start --foreground \
                    --model-store {model_store} \
                    --models {model_name}={model_mar} \
                    --ts-config {ts_config_file}"
                    
    if mar_file.exists():
        subprocess.run(shlex.split(launch_cmd))
    else:
        typer.echo(f"mar file : {mar_file.resolve().__str__()} doesn't exist")
        raise typer.Exit()

#docker run --rm -it --gpus all -p 8080:8080 -p 8081:8081 --name mar \
# -v ${PWD}/model_store:/home/model-server/model-store  \
# -v ${PWD}/config.properties:/home/model-server/config.properties  \
# pytorch/torchserve:latest-gpu torchserve --start --model-store model-store \
# --models resnet18=resnet18.mar --ts-config config.properties
@app.command()
def launch_docker_server(
    model_name:str=typer.Option('resnet18', "-mn", help="Model name"),
    model_store:str=typer.Option('model_store', "-ms", help="Model Store location"),
    ts_config_file:str=typer.Option('config.properties', "-ts", help="TS config file"),
    docker_image:str=typer.Option('pytorch/torchserve:latest-cpu', "-di", help="docker image")
) -> None:
    subprocess.run(shlex.split("rm -rf ./logs"))
    mar_file = Path(f"{model_store}/{model_name}.mar")
    
    cwd = Path('.').resolve().__str__()

    docker_cmd = f"docker run --rm -it -p 8080:8080 -p 8081:8081 --name tsrv  \
                    -v {cwd}/model_store:/home/model-server/model_store \
                    -v {cwd}/config.properties:/home/model-server/config.properties \
                    763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.10.0-cpu-py38-ubuntu20.04-e3 "
                    #pytorch/torchserve:latest-cpu "

    launch_cmd = docker_cmd + f"torchserve --start --foreground \
                    --model-store /home/model-server/{model_store} \
                    --models {model_name}={mar_file.name} \
                    --ts-config /home/model-server/{ts_config_file}"
                    
    if mar_file.exists():
        subprocess.run(shlex.split(launch_cmd))
    else:
        typer.echo(f"mar file : {mar_file.resolve().__str__()} doesn't exist")
        raise typer.Exit()

@app.command()
def regen_launch(
    model_name:str=typer.Option('resnet18', "-mn", help="Model name"),
    model_info_dir:Path=typer.Option('./model_info', "-mm", help="Model .pth directory"),
    extra_files:str=typer.Option('FVHandler.py',"-ef", help="Extra Files"),
    handler:str=typer.Option('serve_handle.py', "-hd", help="Handler function"),
    model_store:str=typer.Option('model_store', "-ms", help="Model Store location"),
    ts_config_file:Optional[str]=typer.Option(None, "-ts", help="TS config file"),
    batch_size:int=typer.Option(1, "-bs", help="batch size"),
    min_workers:int=typer.Option(1, "-minw", help="minimum workers"),
    max_workers:int=typer.Option(None, "-maxw", help="maximum workers")
) -> None:

    mar_file_path = regen_mar(
        model_name=model_name,
        model_info_dir=model_info_dir,
        model_store=model_store,
        extra_files=extra_files,
        handler=handler
    )
    
    model_mar = f"{model_name}.mar"
    if not Path(f"{model_store}/{model_mar}").exists():
        typer.echo(f"Missing {model_mar} in {model_store} directory")
        typer.Exit()
    
    if ts_config_file is None:
        set_config_properties(
            batch_size=batch_size,
            min_workers=min_workers,
            max_workers=max_workers or min_workers
        )
        ts_config_file = 'config.properties'
    else:
        ts_config_file = ts_config_file.name
        
    logger.info(f"Config file used: {ts_config_file}")

    launch_docker_server(
        model_name=model_name,
        model_store=model_store,
        ts_config_file=ts_config_file
    )

if __name__ == '__main__':
    app()