# Use the official Rust image as the base image
FROM rust:1.68 as builder

# Create a directory for the application
WORKDIR /app

# Copy the Cargo.toml and Cargo.lock files
COPY Cargo.toml ./
COPY Cargo.lock ./

# Cache deps download
RUN cargo fetch
# Create a dummy main.rs file to cache dependencies
RUN mkdir src && \
    echo "fn main() {println!(\"Dummy main\");}" > src/main.rs && \
    cargo build --release && \
    rm -rf src

# Copy the real source files
COPY src src

# Build the release version of the application
RUN cargo build --release

# Start a new stage to create the final image
FROM debian:buster-slim

# Install dependencies
RUN apt-get update && \
    apt-get install -y sqlite3 libssl1.1 ca-certificates libstdc++6 wget gdb && \
    rm -rf /var/lib/apt/lists/*

# Copy the binary from the builder stage
COPY --from=builder /app/target/release/breakfast-embed /app/breakfast

# Copy the ONNX model
COPY onnx /app/onnx

# Download and extract ONNX Runtime library
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.8.1/onnxruntime-linux-x64-1.8.1.tgz && \
    tar -xzf onnxruntime-linux-x64-1.8.1.tgz && \
    rm onnxruntime-linux-x64-1.8.1.tgz

# Set the library path to include ONNX Runtime library
ENV LD_LIBRARY_PATH="/onnxruntime-linux-x64-1.8.1/lib:${LD_LIBRARY_PATH}"

# Create a user and group for the application
RUN groupadd -r breakfast && useradd -r -g breakfast breakfast

# Set the user and group
RUN chown -R breakfast:breakfast /app
USER breakfast

# Set the working directory
WORKDIR /app

# Expose the application's port
EXPOSE 8080

# Run the application
CMD ["./breakfast"]
