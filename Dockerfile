# First stage: build the application
FROM clux/muslrust:stable AS builder

WORKDIR /app

# Copy Cargo.toml and Cargo.lock
COPY Cargo.toml Cargo.lock ./

# Create the src directory and a dummy source file to build the dependencies
RUN mkdir src && \
    echo 'fn main() { println!("Dummy build."); }' > src/main.rs

# Cache dependencies
RUN cargo build --release --target=x86_64-unknown-linux-musl
RUN rm -rf src target/x86_64-unknown-linux-musl/release/deps/breakfast*

# Copy the real source files
COPY src src

# Build the application
RUN cargo build --release --target=x86_64-unknown-linux-musl

# Second stage: create the runtime image
FROM alpine:3.15

# Install dependencies
RUN apk add --no-cache sqlite-libs ca-certificates libssl1.1

# Create a user and group for the application
RUN addgroup -S breakfast && adduser -S breakfast -G breakfast

# Create a directory for the application
WORKDIR /app

# Copy the built binary from the first stage
COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/breakfast /app/breakfast

# Set the user and group
RUN chown -R breakfast:breakfast /app
USER breakfast

# Expose the application's port
EXPOSE 8080

# Run the application
CMD ["./breakfast"]
