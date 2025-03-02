FROM ubuntu:24.04
WORKDIR /workdir

RUN apt-get -qq --yes update 

# Rust:

# Install RUST:
RUN apt-get install -y \
    build-essential \
    curl

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

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
RUN source "$SDKMAN_DIR/bin/sdkman-init.sh" && sdk install java 23.0.2-tem
RUN source "$SDKMAN_DIR/bin/sdkman-init.sh" && sdk install scala 3.6.3
RUN source "$SDKMAN_DIR/bin/sdkman-init.sh" && sdk install sbt 1.10.7
RUN source "$SDKMAN_DIR/bin/sdkman-init.sh" && sdk install scalacli 1.6.2

# Install requirements for scala native:
RUN apt -qq --yes install clang libunwind-dev

# Build scala application:
WORKDIR /workdir
ADD scalalang ./scalalang
ENV PATH="${PATH}/:/root/.sdkman/candidates/java/current/bin"
ENV PATH="${PATH}/:/root/.sdkman/candidates/scala/current/bin"
ENV PATH="${PATH}/:/root/.sdkman/candidates/sbt/current/bin"
ENV PATH="${PATH}/:/root/.sdkman/candidates/scalacli/current/bin"
RUN echo $PATH
RUN cd ./scalalang && sbt clean test assembly

# Build scala-native application:
ENV SCALANATIVE_MODE="release-full"
RUN cd ./scalalang && sbt -DNATIVE nativeLink

# Golang:
RUN apt-get install -y golang-go
ADD golang ./golang
RUN cd ./golang && go test ./... && go build

# compile natively prepare_data script:
ADD performance/prepare_data.sc ./performance/prepare_data.sc
RUN cd ./performance && scala-cli --power package --native prepare_data.sc -o prepare_data -f

#
# Build final image
#
FROM eclipse-temurin:23
WORKDIR /root/

# Copy QQWING from Website:
ADD https://qqwing.com/qqwing-1.3.4.jar ./

RUN apt-get -qq --yes update

# Install time/mem tracking tool:
RUN apt -qq --yes install time

WORKDIR /root/

# Move all assembly into ./
COPY --from=0 /workdir/rustlang/target/release/sudoku                         ./sudoku-rust
COPY --from=0 /workdir/scalalang/target/scala-3.6.3/sudoku-assembly-1.0.2.jar ./sudoku-scala.jar
COPY --from=0 /workdir/scalalang/target/scala-3.6.3/sudoku                    ./sudoku-scalanative
COPY --from=0 /workdir/golang/golang                                          ./sudoku-golang
COPY --from=0 /workdir/performance/prepare_data                               ./prepare_data
