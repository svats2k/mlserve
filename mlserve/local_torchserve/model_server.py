from pathlib import Path
import subprocess, shlex

import typer

app = typer.Typer(name="Launch torchserve server", add_completion=False)

#torch-model-archiver --model-name resnet18 \
#--version 1.0 \
#--serialized-file resnet18.pth \
#--extra-files ./FVHandler.py \
#--handler ./serve_handle.py  \
#--export-path model_store -f
@app.command()
def regen_mar(
    model_name:str=typer.Option('resnet18', "-mn", help="Model name"),
    model_pth:str=typer.Option('../models/resnet18.pth', "-mm", help="Model .pth file"),
    extra_files:str=typer.Option('./FVHandler.py',"-ef", help="Extra Files"),
    handler:str=typer.Option('./serve_handle.py', "-hd", help="Handler function"),
    model_store:str=typer.Option('model_store', "-ms", help="Model Store location"),
) -> None:
    
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

#torchserve --start --foreground --model-store model_store/ \
# --models resnet18=resnet18.mar --ts-config config.properties
@app.command()
def launch_server(
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
    model_mar:str=typer.Option('resnet18.mar', "-mm", help="Model .mar file"),
    model_store:str=typer.Option('model_store', "-ms", help="Model Store location"),
    ts_config_file:str=typer.Option('config.properties', "-ts", help="TS config file")
) -> None:
    subprocess.run(shlex.split("rm -rf ./logs"))
    mar_file = Path(f"{model_store}/{model_mar}")
    
    cwd = Path('.').resolve().__str__()

    docker_cmd = f"docker run --rm -it -p 8080:8080 -p 8081:8081 --name tsrv  \
                    -v {cwd}/model_store:/home/model-server/model_store \
                    -v {cwd}/config.properties:/home/model-server/config.properties \
                    pytorch/torchserve:latest-cpu "

    launch_cmd = docker_cmd + f"torchserve --start --foreground \
                    --model-store {model_store} \
                    --models {model_name}={model_mar} \
                    --ts-config {ts_config_file}"
                    
    if mar_file.exists():
        subprocess.run(shlex.split(launch_cmd))
    else:
        typer.echo(f"mar file : {mar_file.resolve().__str__()} doesn't exist")
        raise typer.Exit()

@app.command()
def regen_launch(
    model_name:str=typer.Option('resnet18', "-mn", help="Model name"),
    model_pth:str=typer.Option('../models/resnet18.pth', "-mm", help="Model .pth file"),
    model_mar:str=typer.Option('resnet18.mar', "-mm", help="Model .mar file"),
    extra_files:str=typer.Option('./FVHandler.py',"-ef", help="Extra Files"),
    handler:str=typer.Option('./serve_handle.py', "-hd", help="Handler function"),
    model_store:str=typer.Option('model_store', "-ms", help="Model Store location"),
    ts_config_file:str=typer.Option('config.properties', "-ts", help="TS config file")
) -> None:
    regen_mar(
        model_name=model_name,
        model_pth=model_pth,
        model_store=model_store,
        extra_files=extra_files,
        handler=handler
    )
    
    launch_docker_server(
        model_name=model_name,
        model_mar=model_mar,
        model_store=model_store,
        ts_config_file=ts_config_file
    )

if __name__ == '__main__':
    app()