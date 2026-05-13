FROM gcr.io/oss-fuzz-base/base-builder-rust
RUN apt-get update && apt-get install -y make autoconf automake libtool
COPY . $SRC/neurophone
WORKDIR $SRC/neurophone
COPY .clusterfuzzlite/build.sh $SRC/
