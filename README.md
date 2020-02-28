
# anaconda-tf2
Small framework to train cnn models using tensorflow 2.0 and anaconda on OSCAR

## It is recommended to read the Oscar user manual at [https://docs.ccv.brown.edu/oscar/](https://docs.ccv.brown.edu/oscar/)

## Setting up anaconda

 - Open a terminal Load the anaconda module 
 
    `module load anaconda/3-5.2.0 `
 
    or better add `module load anaconda/3-5.2.0` to the file ` ~/.modules`
 
 - Create the conda environment
 
      `conda create --name tf-gpu python=3.6 tensorflow-gpu pandas matplotlib pillow keras numpy scikit-learn ipykernel`
  
 - Add conda to bashrc  
 
    `echo ". /gpfs/runtime/opt/anaconda/3-5.2.0/etc/profile.d/conda.sh" >> ~/.bashrc`
 
 - Restart terminal
 
 - Activate environment and create a kernel for jupyter notebook 
 
    `notebook conda activate tf-gpu`
 
    `python -m ipykernel install --user --name tf-gpu --display-name "Python (tf-gpu)"`

## Running the scripts
### Start interactive session on a gpu-node
 - First request a gpu-node with 1 CPU 16g of memory and 1 GPU for 3 hours 
 
    `interact -n 1 -t 03:00:00 -m 16g -q gpu -g 1`

- Load the anaconda module if not in the .modules file 

    `module load anaconda/3-5.2.0`

 - Activate the conda environment
 
    `conda activate tf-gpu`

#### Running the jupyter notebooks
 - Open a firefox browser in the background
 
    `firefox &`
    
 - Start the jupyter notebook
 
    `jupyter notebook`

#### Running the scipts 

 - Modify scripts the example [cnnmodel_example.py](https://github.com/brown-ccv/anaconda-tf2/blob/master/scripts/cnnmodel_example.py "cnnmodel_example.py") or [cnnmodel_example_folder.py](https://github.com/brown-ccv/anaconda-tf2/blob/master/scripts/example/cnnmodel_example_folder.py "cnnmodel_example_folder.py")
 
 - Run the script with either 
 
    `python main_process.py cnnmodel_example.py` 

      or 

    `python main_process.py example.cnnmodel_example_folder.py`

## Scheduling a run on Oscar

 - Modify [batch_example.sh](https://github.com/brown-ccv/anaconda-tf2/blob/master/scripts/batch_example.sh "batch_example.sh") to match your job
 - Queue the job 
 
    `sbatch batch_example.sh`

