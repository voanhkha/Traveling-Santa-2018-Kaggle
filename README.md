# Traveling-Santa-2018-Kaggle
The full code of the team rank 8/1874 [Zidmie | Kha | Marc | Simon] in the "Traveling Santa 2018 - Prime Paths" competition on Kaggle (https://www.kaggle.com/c/traveling-santa-2018-prime-paths/leaderboard)

The python code is designed to work much faster with pypy3 (https://pypy.org/download.html). However, CPython can also run this code normally.  
1. Optimize from LKH output tour. We have put in an LKH tour (1502605 raw, 1516256 prime) as the input for step 1. Just run
`pypy3 step1.py` 

2. Kicking (breaking) and fixing the tour iteratively. Whenever the best tour is found, it will be put into folder "tours". The program will take the best tour from the folder after each kicking batch and do the job. Just run
`pypy3 step2.py`  

3. Shuffling. This uses kotlin language (https://kotlinlang.org/). After installing, compile the code by
`kotlinc santa_shuffle.kt -include-runtime -d santa.jar`. Then run `java -jar santa.jar`. Please put the file `cities.csv` into this folder before running, and specify the folder `tours` correctly in variable `dropboxdirstring`

4. Notes: Step 1 can be tuned by inserting more `penalty` steps. Step 2 can be tuned by varying `kick_stride`, `kick_rounds`, and `kick_loss`. Step 3 can be tuned by varying `stride` and `maxfringe`. There is also `EAX.py` which is a genetic algorithm (Edge Assembly Crossover) to combine two tours. 

Team [Zidmie | Kha | Marc | Simon]:  
Yves Miehe <zidmie@free.fr>  
Kha Vo <khahuras@gmail.com>  
Marc Venturini <marc.venturini@gmail.com>  
Simon Dagnarus <simianware@gmail.com>  
