for lang in amh fra hau ibo lin pcm run swa yor
#for lang in amh
do
    for sample in 5 10 20 50
    #for sample in 5

    do
        sbatch job_ipet.sh $lang $sample
    done

done