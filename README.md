# Project Name

LC50 Predictor for QSAR Fish Model

## Demonstration

This project is hosted on AWS and can be accessed using the link below
[AWS](http://52.33.75.17:8501)

## Usage

Two ways to test the predictions.
1. Give single set of values using text boxes and sliders for `CIC0`, `SM1_Dz(Z)`, `GATS1i`, `NdsCH`, `NdssC` and `MLOGP`.
2. Upload a .CSV or .XLS or .XLSX file with any number of rows of data.

## Understanding input values

1. `CIC0` - information indices
2. `SM1_Dz(Z)` - 2D matrix based descriptors
3. `GATS1i` -  2D autocorrelations
4. `NdsCH` - atom type counts
5. `NdssC` - atom type counts
6. `MLOGP` - molecular properties

## Understanding target

`LC50` - Lethal concentration 50 (LC50) is the amount of a substance suspended in the water required to kills 50% of a test animals during a predetermined observation period. LC50 values are frequently used as a general indicator of a substance's acute toxicity.

## Relevant reads

1. QSAR fish toxicity Data Set [UCI](https://archive.ics.uci.edu/ml/datasets/QSAR+fish+toxicity)
2. QSAR investigation of a large data set for fish, algae and Daphnia toxicity [ResearchGate](https://www.researchgate.net/publication/8061505_QSAR_investigation_of_a_large_data_set_for_fish_algae_and_Daphnia_toxicity)
3. How to Deploy a Streamlit App using an Amazon Free ec2 instance? [Medium](https://towardsdatascience.com/how-to-deploy-a-streamlit-app-using-an-amazon-free-ec2-instance-416a41f69dc3)
4. Deploying ML web apps with Streamlit, Docker and AWS [Medium](https://medium.com/usf-msds/deploying-web-app-with-streamlit-docker-and-aws-72b0d4dbcf77)
5. How to write Cron jobs on Amazon Web Services(AWS) EC2 server? [Cumulations](https://www.cumulations.com/blogs/37/How-to-write-Cron-jobs-on-Amazon-Web-ServicesAWS-EC2-server)
