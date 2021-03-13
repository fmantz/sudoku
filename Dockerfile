FROM hseeberger/scala-sbt:8u265_1.4.2_2.11.12

#Copy QQWING from Website:
ADD https://qqwing.com/qqwing-1.3.4.jar ./

RUN apt-get -qq --yes update 

#Install time/mem tracking tool:
RUN apt -qq --yes install time

#Rust:

#Install RUST:
RUN apt -qq --yes install rustc cargo

#Build rust application
ADD rustlang ./rustlang
RUN cd ./rustlang && cargo clean
RUN cd ./rustlang && cargo test 
RUN cd ./rustlang && cargo build --release

#Scala:

#Install requirements for scala native:
#DISABLED: only useful for versions < 0.3 (!)
#RUN apt -qq --yes install clang libunwind-dev 

#Build scala application:
ADD scalalang ./scalalang
RUN cd ./scalalang && sbt clean test assembly

#Build scala-native application:
#DISABLED: only useful for versions < 0.3 (!)
#RUN cd ./scalalang && sbt -DNATIVE nativeLink

#Move all assembly into ./
RUN mv ./rustlang/target/release/sudoku ./sudoku-rust
RUN mv ./scalalang/target/scala-2.11/sudoku-assembly-0.6.0.jar ./sudoku-scala.jar
#DISABLED: only useful for versions < 0.3 (!)
#RUN mv ./scalalang/target/scala-2.11/sudoku-out ./sudoku-scalanative
