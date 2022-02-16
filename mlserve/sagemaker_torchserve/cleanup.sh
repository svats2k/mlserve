docker ps -a | awk 'NR>1 {print $1}' | xargs -I{} docker stop {}
docker ps -a | awk 'NR>1 {print $1}' | xargs -I{} docker rm {}

sudo rm -rf /tmp/tmp*