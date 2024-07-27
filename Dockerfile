FROM ubuntu:22.04
WORKDIR /workdir

RUN apt-get -qq --yes update 

# Rust:

# Install RUST:
RUN apt -qq --yes install rustc cargo

# Build rust application
ADD rustlang ./rustlang
RUN cd ./rustlang && cargo clean
RUN cd ./rustlang && cargo test 
RUN cd ./rustlang && RUSTFLAGS="-C target-cpu=native" cargo build --release

# Scala:
SHELL ["/bin/bash", "-c"]
ENV SDKMAN_DIR=/root/.sdkman

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
RUN source "$SDKMAN_DIR/bin/sdkman-init.sh" && sdk install java 21.0.2-tem
RUN source "$SDKMAN_DIR/bin/sdkman-init.sh" && sdk install scala 3.4.2
RUN source "$SDKMAN_DIR/bin/sdkman-init.sh" && sdk install sbt 1.3.8

# Install requirements for scala native:
RUN apt -qq --yes install clang libunwind-dev

# Build scala application:
WORKDIR /workdir
ADD scalalang ./scalalang
ENV PATH="${PATH}/:/root/.sdkman/candidates/java/current/bin"
ENV PATH="${PATH}/:/root/.sdkman/candidates/scala/current/bin"
ENV PATH="${PATH}/:/root/.sdkman/candidates/sbt/current/bin"
RUN echo $PATH
RUN cd ./scalalang && sbt clean test assembly

# Build scala-native application:
ENV SCALANATIVE_MODE="release-full"
RUN cd ./scalalang && sbt -DNATIVE nativeLink

# Golang:
RUN apt-get install -y golang-go
ADD golang ./golang
RUN cd ./golang && go test ./... && go build

#
# Build final image
#
FROM eclipse-temurin:21
WORKDIR /root/

# Copy QQWING from Website:
ADD https://qqwing.com/qqwing-1.3.4.jar ./

RUN apt-get -qq --yes update

# Install time/mem tracking tool:
RUN apt -qq --yes install time

# install scala to run the scala shell script prepare_data.sh:
SHELL ["/bin/bash", "-c"]
ENV SDKMAN_DIR=/root/.sdkman

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
RUN source "$SDKMAN_DIR/bin/sdkman-init.sh" && sdk install scalacli 1.4.1
ENV PATH="${PATH}/:/root/.sdkman/candidates/scala/current/bin"
RUN echo $PATH

WORKDIR /root/

# Move all assembly into ./
COPY --from=0 /workdir/rustlang/target/release/sudoku                         ./sudoku-rust
COPY --from=0 /workdir/scalalang/target/scala-3.4.2/sudoku-assembly-1.0.0.jar ./sudoku-scala.jar
COPY --from=0 /workdir/scalalang/target/scala-3.4.2/sudoku                    ./sudoku-scalanative
COPY --from=0 /workdir/golang/golang                                          ./sudoku-golang
