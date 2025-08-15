import modal

app = modal.App("example-get-started")


@app.function(gpu="A100")
def test(x):
    print("This code is running on a remote worker!")
    return x**2


@app.local_entrypoint()
def main():
    print("the square is", test.remote(42))