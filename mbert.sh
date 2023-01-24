for i in amh eng fra ibo lin swa yor
#for i in amh

do
	
	for headline_use in only_headline with_text
	do
		for j in 1 2 3 4 5
		#for j in 1

		do
			sbatch run.sh $j $i $headline_use
		done
	done
done