FROM ubuntu:22.10
WORKDIR /workdir

RUN apt-get -qq --yes update 

#Rust:

#Install RUST:
RUN apt -qq --yes install rustc cargo

#Build rust application
ADD rustlang ./rustlang
RUN cd ./rustlang && cargo clean
RUN cd ./rustlang && cargo test 
RUN cd ./rustlang && RUSTFLAGS="-C target-cpu=native" cargo build --release

#Scala:
SHELL ["/bin/bash", "-c"]
ENV SDKMAN_DIR /root/.sdkman

RUN apt-get update && apt-get install -y zip curl
RUN curl -s "https://get.sdkman.io" | bash
RUN chmod a+x "$SDKMAN_DIR/bin/sdkman-init.sh"

RUN set -x \
    && echo "sdkman_auto_answer=true" > $SDKMAN_DIR/etc/config \
    && echo "sdkman_auto_selfupdate=false" >> $SDKMAN_DIR/etc/config \
    && echo "sdkman_insecure_ssl=false" >> $SDKMAN_DIR/etc/config

WORKDIR $SDKMAN_DIR
RUN [[ -s "$SDKMAN_DIR/bin/sdkman-init.sh" ]] && source "$SDKMAN_DIR/bin/sdkman-init.sh" && exec "$@"

RUN source /root/.bashrc
RUN source "$SDKMAN_DIR/bin/sdkman-init.sh" && sdk install java 11.0.16-tem
RUN source "$SDKMAN_DIR/bin/sdkman-init.sh" && sdk install scala 2.13.10
RUN source "$SDKMAN_DIR/bin/sdkman-init.sh" && sdk install sbt 1.3.8

#Install requirements for scala native:
#DISABLED: only useful for versions < 0.3 (!)
#RUN apt -qq --yes install clang libunwind-dev 

#Build scala application:
WORKDIR /workdir
ADD scalalang ./scalalang
ENV PATH="${PATH}/:/root/.sdkman/candidates/java/current/bin"
ENV PATH="${PATH}/:/root/.sdkman/candidates/scala/current/bin"
ENV PATH="${PATH}/:/root/.sdkman/candidates/sbt/current/bin"
RUN echo $PATH
RUN cd ./scalalang && sbt clean test assembly

#Build scala-native application:
#DISABLED: only useful for versions < 0.3 (!)
#RUN cd ./scalalang && sbt -DNATIVE nativeLink

#
# Build final image
#
FROM eclipse-temurin:11
WORKDIR /root/

#Copy QQWING from Website:
ADD https://qqwing.com/qqwing-1.3.4.jar ./

RUN apt-get -qq --yes update

#Install time/mem tracking tool:
RUN apt -qq --yes install time

#Move all assembly into ./
COPY --from=0 /workdir/rustlang/target/release/sudoku                        ./sudoku-rust
COPY --from=0 /workdir/scalalang/target/scala-2.13/sudoku-assembly-0.9.0.jar ./sudoku-scala.jar
#DISABLED: only useful for versions < 0.3 (!)
#RUN mv ./scalalang/target/scala-2.11/sudoku-out ./sudoku-scalanative
