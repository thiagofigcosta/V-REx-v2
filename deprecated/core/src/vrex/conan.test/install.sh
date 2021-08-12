#!/bin/bash

sudo apt install build-essential

git clone https://github.com/conan-io/conan.git /home/$USER/.conan_files

pip 1>/dev/null 2>&1
if [ $? -eq 0 ] ; then
    pip="pip"
    python="python"
    cd /usr/bin
    sudo mv python python.bkp 1>/dev/null 2>&1
    sudo ln -s /usr/bin/python2.7 /usr/bin/python
else
    pip3 1>/dev/null 2>&1
    if [ $? -eq 0 ] ; then
        pip="pip3"
        python="python3"
    fi
fi

if [ -z $pip ] ; then
    echo "pip command not found, install it"
    exit 1
else
    echo "Found pip: $pip"
fi

cd /home/$USER/.conan_files

eval $pip install -r conans/requirements.txt

printf "#/usr/bin/env $python\n\nimport sys\n\nconan_repo_path = \"/home/$USER/.conan_files\" # ABSOLUTE PATH TO CONAN REPOSITORY FOLDER\n\nsys.path.append(conan_repo_path)\nfrom conans.client.command import main\nmain(sys.argv[1:])\n" > /home/$USER/.conan_files/conan_run.py
printf "#/bin/bash\n$python /home/$USER/.conan_files/conan_run.py \$@\n" > /home/$USER/.conan_files/conan.sh


chmod +x /home/$USER/.conan_files/conan.sh
sudo ln -s /home/$USER/.conan_files/conan.sh /usr/bin/conan

conan remote add huang https://api.bintray.com/conan/huangminghuang/conan 
