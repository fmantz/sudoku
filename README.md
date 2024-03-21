# Sudoku

By chance, I watched a YouTube video on backtracking [https://www.youtube.com/watch?v=G_UYXzGuqvM] (Python Sudoku Solver - Computerphile).
I decided it would be a nice toy project to develop it in my favorite programming language **Scala**, 
and **compare the code, performance and memory usage** afterwards with another programming language I would like to learn, **Rust**.

It was actually **NOT** my goal to develop the fastest Sudoku solver possible (!), 
rather than comparing these two programming languages **myself**.
I will improve the performance 'maybe' in a later version 0.2. There are lots of possibilities. 
However, if you look for a really fast Sudoku solver try:

* https://qqwing.com or
* https://github.com/Emerentius/sudoku

My approach was:

* I developed a Scala version first. I did not use any third party libraries. I didn't think about JVM specific performance bottlenecks like 'array 2ds are slow'.
* Afterwards I developed a version in Rust. Since this is my first Rust program the code may look a bit awkward to Rust developers, I apologize. I tried to copy the Scala code as much as possible. There is for every method / test in the Scala code a corresponding method / test in the Rust code.   
* I measured the performance (wall time, peak memory) with a unix tool **/usr/bin/time** (note, it is **NOT** the shell 'time' command).

## Build:

To easily try it yourself, I added a Docker-build file to the project. Use the tags to choose one of the available versions (it might be that you need to update the base image to a newer version):

```bash
git checkout tags/version-1.0.0 -b v1.0.0
docker build . --tag sudoku:1.0.0
```

This Docker build will:

* Download QQWing from [https://qqwing.com] for generating Sudoku puzzles. It's a very fast Sudoku solver (and generator) using a more advanced algorithm.
* Install Rust and compile the Rust version of the program.
* Install Scala NATIVE requirements and build two assemblies, one JAR and Scala NATIVE one (until Version 0.2). Scala is already installed in the base image. 

Note, the Docker image will be around **2 GB**.

## Run

```bash
docker-compose up
```

This Docker run will:

1. Create a subdirectory in folder ./performance with a timestamp.
2. Generate x Sudokus for each level (Level 1 = 1 Sudoku, Level 2 = 10 Sudokus, Level 3 = 100 Sudokus ...). Four levels are currently set. 
  Note, the number of levels can be changed in file **[./performance/test.sh](./performance/test.sh)**.
3. First, solve all Sudokus of the current level very fast with QQWing.
4. Second, solve all Sudokus with the Rust version of my program.
5. Third, solve all Sudokus with the Scala JAR version of my program.
6. Fourth, solve all Sudokus with the Scala NATIVE version of my program (version <= Version 0.2). 
7. Create two CSVs files `mem.csv` (in kb) and `time.csv` collecting the current performance measures from the log-files.
8. Continue with Step 2 to process the next level until Level 4.

## Results

* The Scala code is more comprehensive than the Rust code but much less then I first thought. 
* I did not expect such a big difference but the Rust program is much faster then both Scala versions, also the peak memory is muss less then the Scala JAR version.
* The startup time of the Scala NATIVE version is a bit faster then the Scala JAR version but the overall performance is the opposite when solving many Sudokus.
* However, the memory consumption of the Scala NATIVE version is much lower than the Scala JAR version. 

I put the results of my test with 6 levels into folder [./performance/version_0.1-result](./performance/version_0.1-result). I run it in Docker on my local linux machine:

```bash
OS: Manjaro Linux x86_64 
Host: Precision T3600 01 
Kernel: 5.15.85-1-MANJARO 
CPU: Intel Xeon E5-4650L 0 (16) @ 3.100GHz 
GPU: NVIDIA Quadro P620 
Memory: 4076MiB / 64262MiB
```
Used programming language versions:

* Scala 3.4.0 (on OpenJDK 21)
* Rust Edition 2021

## Run manually

Commands can be manually run by:

```bash
docker container run -it --name sudoku sudoku:1.0 bash
```

The **/root** directory (also current directory) will contain all command line programs:

* qqwing-1.3.4.jar  
* sudoku-rust  
* sudoku-scala.jar  
* sudoku-golang

## Update: Version 0.2

Since my sudoku program was a bit slow, I thought about speeding it up a bit. Therefore, I developed a 'turbo' for my algorithm which does two things:

* Speedup the check if the conditions are satisfied by using precomputed bitsets.
* Starting to search a solution in the rows and columns having the most entries first. 

These two changes improved the speed of the programs a lot. Nevertheless, the speed of the Rust version was still around factor 4 times faster than the Scala version. 
Unsurprisingly, QQWing is still the fastest, however I 'may' try to use multithreading in a next version. QQWing is using multithreading but also a much more elaborate algorithm.
The Scala JAR version was still a bit faster than the Scala NATIVE version. The low memory consumption of the Rust and Scala NATIVE version was best. Not having a JVM is in this 
respect a real advantage. In a Version 0.3 Scala NATIVE would not be available anymore, since it still does not have multithreading support. 
I put the results of my test with 6 levels into folder [./performance/version_0.2-result](./performance/version_0.2-result).

## Update: Version 0.3

Even I have still new ideas, this will probably be my last version of my Sudoku implementation. This time I added multithreading.
Again, my main motivation was to learn a bit Rust (I am still a bloody beginner) and compare Scala with Rust. Again, I put my results in an own folder.
However, this time I added two sub directories. 

The first folder [./performance/version_0.3-result/cpu-1](./performance/version_0.3-result/cpu-1) contains the results with my usual resources restrictions on Docker (see extract from docker-compose.yml below). 
The other folder contains the results without those restrictions [./performance/version_0.3-result/cpu-all](./performance/version_0.3-result/cpu-all). 
We see, the performance still not match the one of QQWing but I got better. I would need to switch the algorithms to really compete, anyway this was not my motivation.  

```bash
        resources: 
          limits:
            cpus: '1.00' 
          reservations:
            cpus: '1.00'
```
I also added a new folder two compare all results of **my** own implementations [./performance/compare_v1-v3](./performance/compare_v1-v3).
We see the performance improved a lot between Version 1 and Version 3. My Sudoku Rust implementations have been consistently around factor 4 faster than the Scala implementations.
The next picture shows the time spend to solve 100000 Sudokus.

![Time to solve 100000 Sudokus](./performance/compare_v1-v3/time.png)

Also the memory consumption was better in the Rust versions than in the Scala versions but Scala NATIVE was not much worse. 
The next picture shows the memory used to solve 100000 Sudokus (in kb).

![Peak memory usage in kb to solve 100000 Sudokus](./performance/compare_v1-v3/mem.png)

Even Scala has other advantages (like it is less low level), I maybe should consider to work with Rust in the future. At least in some cases, it was fun :-). 

## Update: Version 0.4

Finally, I made two changes. However, the performance did not improve. The speed of the Rust version did not change or got minimal faster. The speed of the Scala version got a bit worse.
The memory consumption of the Rust version nearly did not change but the memory consumption of the Scala version improved due to the second change. 
I did following two small changes:

* A saved all possible combinations of numbers to check for an empty slot in a constant so that those number sequences must not be computed on the fly. 
* I saved the Scala array in a 2D array of bytes instead of integers. Earlier, I did not do this change because it did not improve the performance.

Again you can find all results here [./performance/version_0.4-result](./performance/version_0.4-result).

## Update: Version 0.5 (last version using a recursive approach)

I added a new small update. In Version 4.0 I added all possible combinations of numbers ("the powerset"), each as sorted array. 
In Version 0.5, I shuffled those arrays so that not always the same numbers are tried into empty slots first. Therefore, there should be less collisions. 
This improved the performance of the Scala version a bit. However, the performance of the Rust version did not change. 

As usual, you can find the results here [./performance/version_0.5-result](./performance/version_0.5-result).

## Update: Version 0.6 (using an iterative approach)

This time, I wanted to learn again something new. I decided to implement backtracking without recursion. 
In particular I did the following changes:

* I developed the solving approach from scratch non-recursively.
* I choose a one dimensional array to store the puzzle (since this was easier to implement here).
* I sorted the cells for solving differently: I sorted them by number of possiblities available (ascending).

The result was a algorithm that was twice as slow as Version 0.5 (in the Rust Version) and around 30% slower (in the Scala Version). 
This surprised me a bit but may depend on the sorting and extra steps I implemented. The memory consumption did not change so much. 

As usual, you can find the results here [./performance/version_0.6-result](./performance/version_0.6-result).

This version was also considered as pre-step for what I want to learn next: **CUDA**. On GPU, threads have relatively small stacks, so it is best practice to avoid them (according to what I found on the web). 
Nevertheless, I may have used the stack here. The stack for each Sudoku should be bound to 81 elements. This should be small enough, I guess.

## Update: Version 0.7 (getting closer to CUDA)

I implemented this version only to get another step closer to port the algorithm to CUDA. GPUs work differently to CPUs. 
In contrast to CPUs which are optimized for parallel tasks and a low latency, GPUs are optimized for processing data in parallel and a high throughput. 
Therefore, branching is bad on a GPU. For this reason I eliminated branching in the algorithm as far as possible. There are only a few 'if then else' conditions left. In particular I did following changes:

* I removed the fastIndexOf method against another index array.
* I chose smaller datatypes for indices in the Rust version (and added some casts). 

The algorithm became marginal faster in Scala as well as in Rust (I always look at 100000 Sudokus). 
As usual, you can find the results here [./performance/version_0.7-result](./performance/version_0.7-result).

## Update: Version 0.8 (cleanups and CUDA)

In this version, I cleaned the rust code up. I also updated the Rust Edition to 2021. Moreover, I developed a very disappointing CUDA version.
In addition, I used a higher number of parallel tasks. The result is that the RUST program is again as fast as Version 0.5. 
However, the Scala version is still a bit slower than the Scala program Version 0.5.

As usual, you can find the results here [./performance/version_0.8-result](./performance/version_0.8-result).

The algorithm could also be fully implemented with CUDA. However, the CUDA program is very slow. 
I guess few data branches ('ifs') is not enough for a fast CUDA program, also parallel data access is needed (SMID, single-instruction-multiple). 
Comparing the CUDA version with the RUST version manually, (without my docker test), I get following performance differences:

|Sudokus |      CUDA    |        RUST   |
--------:|-------------:|--------------:|
|       1|   0.432938s  |   0.151921058s|
|      10|   0.811146s  |   0.157538849s|
|     100|   2.202813s  |   0.185084571s|
|    1000|  14.222719s  |   0.470379894s|
|   10000| 153.452142s  |   2.769783619s|
|  100000| 948.916880s  |  25.024930099s|

I assume unless I do not have a much better GPU than the CPU, I am better of with solving such problems on the CPU.
Anyway, I learnt something, lets move on, maybe to "golang"? 

## Update: Version 0.9 (golang here we go!)

What have I done? I did some small cleanups in the Rust and Scala versions, added some Rust compiler optimization flags, moved from Scala 2.11 to Scala 2.13 with OpenJdk 11 Eclipse Temurin. 
And finally added a Golang version! It was fun to program Golang, at least the syntax can be picked up quite fast ... To my surprise the Scala version became much faster (more than factor two)
with the new Scala SDK/Java JDK, the memory consumption became worse. Even I added some compiler flags the already fast Rust solver did not change its performance. 
The new Golang Sudoku solver was, as expected, between the Scala and Rust version in respect to performance and memory consumption. Up to 1000 Sudokus the Golang version was 
even the fastest. Golang has a minimum of startup time! However, I assume the Rust version is slower (up to 1000 Sudokus) because of 'Rayon' the parallelization library I had to use since I did not want 
to implement a thread pool myself. Rayon is optimized for throughput not short startup times, maybe Rust async would have been a better choice in this respect. 

As usual, you can find the results here [./performance/version_0.9-result](./performance/version_0.9-result).

What comes next? I don't know. Maybe I will transform the CUDA version to plain C. Did I enjoyed to program Golang more than Rust? I don't know. Both two for me new languages were fun! 
Some concepts I found cleaner in Rust, e.g. I disliked in Golang that I could not create constants arrays. Therefore, I really did not like to work with String manipulation in Rust.
Golang has a nice clean syntax, you need to write less code than in Rust and you pick it up much quicker. And what I really enjoyed was that it has really the batteries included: 
parallelization, formatting, a test framework, benchmark tests, profiling, race condition detection... 
