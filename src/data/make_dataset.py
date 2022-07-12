# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

""" Runs data processing scripts to turn raw data from (../raw) into
       cleaned data ready to be analyzed (saved in ../processed).
"""
logger = logging.getLogger()
logger.info('making final data set from raw data')

#---Choose working directory
root = tk.Tk()
root.withdraw()
directory=filedialog.askdirectory()#user defined directory
logger.info('Define the directory of your data (folder data/raw/')
os.chdir(directory)

#---Import both train and test sets
titanic_train_data=pd.read_csv(directory+"/train.csv",header=0, encoding="utf-8")
titanic_test_data=pd.read_csv(directory+"/test.csv",header=0, encoding="utf-8")


# =============================================================================
# @click.command()
# @click.argument(directory+'/train.csv', type=click.Path(exists=True))
# @click.argument(directory+'/test.csv', type=click.Path())
# def main(input_filepath, output_filepath):
#     """ Runs data processing scripts to turn raw data from (../raw) into
#         cleaned data ready to be analyzed (saved in ../processed).
#     """
#     logger = logging.getLogger(__name__)
#     logger.info('making final data set from raw data')
# 
# 
# if __name__ == '__main__':
#     log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#     logging.basicConfig(level=logging.INFO, format=log_fmt)
# 
#     # not used in this stub but often useful for finding various files
#     project_dir = Path(__file__).resolve().parents[2]
# 
#     # find .env automagically by walking up directories until it's found, then
#     # load up the .env entries as environment variables
#     load_dotenv(find_dotenv())
# 
#     main()
# =============================================================================


