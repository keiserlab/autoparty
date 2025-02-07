singularity run containers/redis.sif & # start redis server
singularity run -B $1:/mnt,$2:/inputs,$3:/outputs containers/autoparty.sif $4 $5 # run autoparty container, starts flask and celery
