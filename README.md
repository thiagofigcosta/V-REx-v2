# V-REx-v2 ICNN
Vulnerabilities' Risk of Exploitation

Improved version of https://github.com/thiagofigcosta/V-REx

Prototype of a genetically tuned neural network to predict probability of exploiting a vulnerability.

Neural network with parts of the code from: 
>- https://github.com/kapilt/mongoqueue

Raw dataset:
>- https://mega.nz/file/H4wzwKpb#ONvpbhR_GDy3vxKEabHZQHqvA-5EM4WWBbwlbDzBTt4

Data sources:
>- https://cve.mitre.org/
>- https://cwe.mitre.org/index.html
>- https://nvd.nist.gov/
>- https://www.exploit-db.com/
>- http://capec.mitre.org/
>- https://oval.cisecurity.org/repository/search
>- https://www.cvedetails.com/

## To run commands

Access the front-end container and use one of the following commands:

>- bash
>- front
>- frontend
>- front-end

Type `h` + enter for help

### Example:
```
docker exec -it $(docker container ls | grep front-end | cut -f 1 -d' ') front
```

## To follow logs

To follow logs type one of the commands below

### Core V2
```
docker logs --follow $(docker container ls | grep core-2 | cut -f 1 -d' ')
```

### Data Crawler
```
docker logs --follow $(docker container ls | grep data-crawler | cut -f 1 -d' ')
```
### Data Processor 
```
docker logs --follow $(docker container ls | grep data-processor | cut -f 1 -d' ')
```

## Passwords location
Inside .env file

## Requirements
>- docker
>- docker-compose

## Images required
>- mongo-express:0.54.0
>- mongo:4.0.20
>- python:3.8
>- python:3.8-slim


## Running with Docker compose (recommended)
```
docker-compose build
docker-compose up
```
or in background
```
docker-compose build
docker-compose up -d
```
or
```
docker-compose up -d --scale portainer=0 --scale data-crawler=0 --scale data-processor=0
```
or in background with replicas
```
docker-compose up -d --scale data-crawler=2
```
or in background without some modules
```
docker-compose up -d --scale data-crawler=0 --scale data-processor=0
```

To Stop run:
```
docker-compose stop
```

## Running with Docker

### Build docker images

#### Data Crawler
```
./update_Pytho\{N\}.sh # update Pytho{\}
docker build -t data-crawler:v1.0.0 data-crawler/  # build image
docker rmi -f $(docker images -f "dangling=true" -q) # cleanup <none> images
```

#### Data Processor
```
./update_Pytho\{N\}.sh # update Pytho{\}
docker build -t data-processor:v1.0.0 data-processor/  # build image
docker rmi -f $(docker images -f "dangling=true" -q) # cleanup <none> images
```

#### Front end
```
./update_Pytho\{N\}.sh # update Pytho{\}
docker build -t front-end:v1.0.0 front-end/  # build image
docker rmi -f $(docker images -f "dangling=true" -q) # cleanup <none> images
```

#### Core V2
```
./update_Pytho\{N\}.sh # update Pytho{\}
docker build -t core:v2.0.0 core-2/  # build image
docker rmi -f $(docker images -f "dangling=true" -q) # cleanup <none> images
```

### Running docker images

#### Data crawler
```
docker run data-crawler:v1.0.0
```

#### Front end
```
docker run front-end:v1.0.0
```

#### Core V2
```
docker run core:v2.0.0
```

## Running locally

To set up virtual env
```
pip3 install virtualenv
python3 -m virtualenv -p python3 venv
```

To activate virtual env
```
source venv/bin/activate
```

Deploy mongo on docker or on host machine then install dependencies and run the application on the activated enviroment!

To deactivate virtual env
```
deactivate
```

To delete venv:
```
rm -rf venv
```f venv
```

## Cleanup docker compose volumes
```
sudo rm -rf docker_volumes
```

## Cleanup RAM memory cache if needed (to free RAM for huge page allocation)
```
sudo su
free -mh && sync && echo 3 > /proc/sys/vm/drop_caches && free -mh
```

## Follow CPU usage
```
top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}'
```
or
```
awk '{u=$2+$4; t=$2+$4+$5; if (NR==1){u1=u; t1=t;} else print ($2+$4-u1) * 100 / (t-t1) "%"; }' \
<(grep 'cpu ' /proc/stat) <(sleep 1;grep 'cpu ' /proc/stat)
```

## Check VSZ usage
```
{ ps -aux | head -n 1 ; ps -aux | grep -E "S|Rl" | grep -v -E "keep-alive" | grep -E "python3 .tmp_pythoN" ;} | awk 'NR>1 {$5=int($5/1024)"M";}{ print;}'
```
or
```
top
```


## To profile memory
Import the lib `from memory_profiler import profile` and put the anotation over the function to be analyzed `@profile`.
Run the profiler, e.g.:
```
mprof run -M Pytho\{N\}.py src/test.py ; mprof plot
```
Or use the location of mprof, something like:
```
python3 ~/.local/lib/python3.6/site-packages/mprof.py run -M Pytho\{N\}.py src/test.py ; python3 ~/.local/lib/python3.6/site-packages/mprof.py plot
```
Or run the code manually, e.g.:
```
python3 -m memory_profiler Pytho\{N\}.py src/test.py
```


## Web Interfaces:
>- Mongo-Express: http://localhost:8081/
>- Portainer: http://localhost:9000/
