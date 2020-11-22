# Sudoku

By chance, I watched a YouTube video on backtracking [https://www.youtube.com/watch?v=G_UYXzGuqvM] (Python Sudoku Solver - Computerphile).
I decided it would be a nice toy project to develop it in my favorite programming language **Scala**, 
and **compare the code, performance and memory usage** afterwards with another programming language I would like to learn, **Rust**.

It was actually **NOT** my goal to develop the fastest sudoku solver possible (!), 
rather than comparing these two programming languages **myself**.
I will improve the performance 'maybe' in a later version 0.2. There are lots of possibilities. 

My approach was:

* I developed a Scala version first. I did not use any third party libraries. I didn't think about JVM specific performance bottlenecks like 'array 2ds are slow'.
* Afterwards I developed a version in Rust. Since this is my first Rust program the code may look a bit awkward to Rust developers, I apologize. My only source to learn Rust was Google. I tried to copy the Scala code as much as possible. There is for every method / test in the Scala code a corresponding method / test in the Rust code.   
* I measured the performance (wall time, peak memory) with a unix tool **/usr/bin/time** (note, it is **NOT** the shell 'time' command).

## Build:

To easily try it yourself. I added a Docker-build file to the project:

```bash
docker build . --tag sudoku:0.1
```

This Docker build will:

* Download QQWing from [https://qqwing.com] for generating sudoku puzzles. It's a very fast sudoku solver using a more advanced algorithm.
* Install Rust and compile the Rust version of the program
* Install Scala NATIVE requirements and build two assemblies, one JAR and Scala NATIVE one. (Scala is already installed in the base image)  

Note, the Docker image will be around **2 GB**.

## Run

```bash
docker-compose up
```

This Docker run will:

1. Create a subdirectory in folder ./performance with a timestamp.
2. Generate x Sudokus for each level (level 1 = 1 sudoku, level 2 = 10 Sudokus, level 3 = 100 Sudokus ...). Four level are currently set. 
  Note, the number of levels can be changed in file **./performance/test.sh**.
3. First, solve all Sudokus of the current level very fast with QQWing.
4. Second, solve all Sudokus with the Rust version of my program.
5. Third, solve all Sudokus with the Scala JAR version of my program.
6. Fourth, solve all Sudokus with the Scala NATIVE version of my program.
7. Create two CSVs files `mem.csv` (in kb) and `time.csv` collecting the current performance measures from the log-files.
8. Continue with step 2 to process the next level until level 4.

## Results

* The Scala code is more comprehensive than the Rust code but much less then I first thought. 
* I did not expect such a big difference but the Rust program is much faster then both Scala versions, also the peak memory is muss less then the Scala JAR version.
* The startup time of the Scala NATIVE version is a bit faster then the Scala JAR version but the overall performance is the opposite when solving many Sudokus.
* However, the memory consumption of the Scala NATIVE version is much lower than the Scala JAR version. 

I put the results of my test with 6 levels into folder ./performance/version_0.1-result. I run it in Docker on my local linux machine:

```bash
OS: Manjaro Linux x86_64 
Host: Precision T3600 01 
Kernel: 5.8.18-1-MANJARO 
CPU: Intel Xeon E5-4650L 0 (16) @ 3.100GHz 
GPU: NVIDIA Quadro 600 
Memory: 7164MiB / 32068MiB 
```
Used programming language versions:

* Scala 2.11.12
* Rust Edition 2018

## Run manually

Commands can be manually run by:

```bash
docker container run -it --name sudoku sudoku:0.1 bash
```

The **/root** directory (also current directory) will contain all command line programs:

* qqwing-1.3.4.jar  
* sudoku-rust  
* sudoku-scala.jar  
* sudoku-scalanative
