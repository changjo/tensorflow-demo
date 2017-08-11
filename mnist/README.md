## MNIST

### Requirements

- scipy, numpy, pillow

For example:

```
$ pip3 install scipy
$ pip3 install numpy
$ pip3 install pillow
```

### How to Use

#### Simple example
```python

import predict_mnist

image_files = ["test_examples/8/1.png", "test_examples/0/1.png", "test_examples/3/1.png"]

predict = predict_mnist.start()
predicted_digits = predict(image_files)

for i in range(len(predicted_digits)):
  print(image_files[i], "->", predicted_digits[i])

```

Results:
```
test_examples/8/1.png -> 8
test_examples/0/1.png -> 0
test_examples/3/1.png -> 3
```

___

`test_examples` 안에 디렉토리 이름은 실제 label이고 각 숫자당 10개씩 (`1.png` ~ `10.png`) 테스트 이미지들이 있습니다.

예)
`test_examples/8/2.png` : 실제 레이블이 8인 이미지.

___
