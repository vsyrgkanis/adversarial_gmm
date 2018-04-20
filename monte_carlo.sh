dir=$1
mkdir -p $dir
echo $dir
for dgp_two in 0;
do
    for samples in 1000;
    do
        for strength in 1;
        do            
            for n_critics in 50;
            do    
                for dimension in 1;
                do
                    for num_steps in 400;
                    do
                        for radius in 50;
                        do
                            for jitter in 1;
                            do
                                for func in 'abs' '2dpoly' 'sigmoid' 'step' '3dpoly' 'sin' 'linear';
                                do
                                    for number in $(seq 1 20); 
                                    do
                                        echo "Iteration ${number}" 
                                        python monte_carlo.py --iteration $number --dir $dir --num_instruments $dimension --n_samples $samples --num_steps $num_steps --func $func --radius $radius --n_critics $n_critics --strength $strength --jitter $jitter --dgp_two $dgp_two
                                    done
                                    python combine.py --dir $dir --num_instruments $dimension --n_samples $samples --num_steps $num_steps --func $func --radius $radius --n_critics $n_critics --strength $strength --jitter $jitter --dgp_two $dgp_two
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
exit 0