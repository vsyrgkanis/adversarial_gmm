dir=$1
mkdir -p $dir
echo $dir
it=0
for dgp_two in 0;
do
    for samples in 1000;
    do
        for strength in .5 .7 .9;
        do            
            for n_critics in 50;
            do    
                for dimension in 1 2 3 4 5 6 7 8 9 10;
                do
                    for num_steps in 400;
                    do
                        for radius in 50;
                        do
                            for jitter in 1;
                            do
                                for func in 'abs' '2dpoly' 'sigmoid' 'step' '3dpoly' 'sin' 'linear' 'rand_pw';
                                do
                                    for number in $(seq 1 100);
                                    do                           
                                        echo $it >> "${dir}/overall_log.txt"
                                        echo "Iteration ${number}" &>> "${dir}/log_${it}.txt"
                                        python monte_carlo.py --iteration $number --dir $dir --num_instruments $dimension --n_samples $samples --num_steps $num_steps --func $func --radius $radius --n_critics $n_critics --strength $strength --jitter $jitter --dgp_two $dgp_two &>> "${dir}/log_${it}.txt"
                                        it=$((it+1))
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done


for dgp_two in 0;
do
    for samples in 1000;
    do
        for strength in .5 .7 .9;
        do            
            for n_critics in 50;
            do    
                for dimension in 1 2 3 4 5 6 7 8 9 10;
                do
                    for num_steps in 1000;
                    do
                        for radius in 50;
                        do
                            for jitter in 1;
                            do
                                for func in 'abs' '2dpoly' 'sigmoid' 'step' '3dpoly' 'sin' 'linear' 'rand_pw';
                                do                          
                                    echo $it >> "${dir}/overall_log.txt"
                                    python combine.py --dir $dir --num_instruments $dimension --n_samples $samples --num_steps $num_steps --func $func --radius $radius --n_critics $n_critics --strength $strength --jitter $jitter  --dgp_two $dgp_two &>> "${dir}/log_${it}.txt"
                                    it=$((it+1))
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done