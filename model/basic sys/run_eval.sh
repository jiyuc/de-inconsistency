for T in mclass
do
    TEST=/model/data/test.txt
    echo $TEST
    python evaluation.py $TEST $T
done
          
