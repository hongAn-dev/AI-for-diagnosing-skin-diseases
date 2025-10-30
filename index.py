import tensorflow as tf  #Nhập thư viện Tensonflow

# Helper libraries
import numpy as np      #Nhập thư viện Numpy liên quan đến mảng đa chiều và ma trận
import matplotlib.pyplot as plt     #Nhập thư viện Matplotlib để vẽ đồ thị và trực quan hóa dữ liệu

print(tf.__version__) #Kiểm tra phiên bản của tensorflow

fashion_mnist = tf.keras.datasets.fashion_mnist #Truy cập dến dataset Fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() #Tải dataset về, đặc biệt là tập Train và tập Test

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'] #List 10 lớp trong dataset Fashion MNIST

train_images.shape #Trả về kích thước của mảng Numpy chứa dữ liệu hình ảnh Training (Số lượng mẫu, Chiều cao, Chiều rộng)

len(train_labels)   #Trả về số lượng phần tử trong mảng Training

train_labels #Danh sách các nhãn số nguyên, xác định đáp án đúng cho mỗi hình ảnh trong tập Training.

test_images.shape  #Trả về kích thước của tập dữ liệu hình ảnh dùng trong Testing

len(test_labels)   #Trả về số lượng phần từ trong mảng Test


#Sử dụng thư viện Matplotlib để hiển thị hình ảnh đầu tiên trong tập Training ( Training.image[0] ) và các giá trị pixel của nó
plt.figure()                    #Khởi tạo cửa sổ đồ họa mới nơi hình ảnh được vẽ
plt.imshow(train_images[0])     #Vẽ ma trận pixel của hình ảnh đầu tiên
plt.colorbar()                  #Hiển thị 1 thanh đo bên cạnh cho thấy mối quan hệ giữa màu sắc trên ảnh và giá trị pixel tương ứng
plt.grid(False)                 #Loại bỏ các đường lưới mặc định
plt.show()                      #Hiển thị hình ảnh



#Chuẩn hóa giá trị của tất cả các pixel trong tập Testing và Training
train_images = train_images / 255.0
test_images = test_images / 255.0
#Tại sao chia cho 255 : Vì mỗi pixel được biểu diễn bằng 1 số nguyên có giá trị từ 0 đến 255 nên chia 255 để đưa về phạm vi mới [0.0 --> 1.0]

plt.figure(figsize=(10,10))     #Tạo ra 1 cửa sổ lớn hơn để chứa được 25 ảnh xét tiếp theo
for i in range(25):             #Lặp từ 0 --> 24
    plt.subplot(5,5,i+1)        #Chia cửa sổ đồ họa thành 5x5=25 và chọn ô i+1 để vẽ hình ảnh hiện tại
    plt.xticks([])              #Ẩn trục tọa độ x
    plt.yticks([])              #Ẩn trục tọa độ y
    plt.grid(False)             #Loại bỏ các đường lưới mặc định
    plt.imshow(train_images[i], cmap=plt.cm.binary)         #Vẽ hình ảnh thứ i và đảm bảo hình ảnh xám được hiển thị với màu trắng đen
    plt.xlabel(class_names[train_labels[i]])                #Thêm nhãn dán cho hình ảnh
plt.show()                      #Hiển thị tất cả


#Xây dựng mạng nơ-ron yêu cầu cấu hình các lớp của mô hình, sau đó biên dịch mô hình
model = tf.keras.Sequential([                               #Khai báo 1 mô hình tuyến tính trong đó dữ liệu đi qua từng lớp một theo thứ tự
    tf.keras.layers.Flatten(input_shape=(28, 28)),          #Hình ảnh là 1 ma trận 2 chiều. Mạng nơ-ron truyền thống chỉ chấp nhận 1 chiều. Flatten là để duối thẳng ma        trận                                                            28x28 thành 1 dãy 28x28=784

    #Dense : Thực hiện phép nhân ma trận và cộng bias
    tf.keras.layers.Dense(128, activation='relu'),          #Siêu tham số được chọn để học các tính năng từ 784 pixel đầu vào
    tf.keras.layers.Dense(10)                               #Lớp đầu ra có 10 nơ-ron
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
#1) Tối ưu hóa : Chịu trách nhiệm điều chỉnh trọng số (weights) của mô hình để giảm thiếu hàm mất mát sau mỗi lần xử lý dữ liệu ___ adam : Thuật toán tối ưu
#2) Hàm mất mát : Đo lường mức độ sai sót giữa dự đoán mô hình và nhãn dãn thực tế  ___ Hàm mất mát tiêu chuẩn cho các bài toán phân loại đa lớp ___ Cho biết đầu ra cuối cùng là các logits ( điểm số thô ) chứ không phải xác suất
#3) Chỉ số đánh giá : Các thước đo để theo dõi hiệu suất mô hình trong quá trình Training và Testing ___ Đo lường tỷ lệ phần trăm của các hình ảnh

model.fit(train_images, train_labels, epochs=10)        #Training the Model
#Dữ liệu đầu vào ___  Nhãn thực tế ___ Số lần lặp lại



#Đánh giá hiệu suất
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)      #Tính toán hàm mất mát ___ verbose=2 --> Hiển thị thanh tiến trình
print('\nTest accuracy:', test_acc)                 


probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])     
#Softmax : Mô hình xác suất ___ Chuyển đổi những logits thành xác suất


predictions = probability_model.predict(test_images)    #Tạo 1 bảng Numpy ( Hàng-Hình ảnh ___ Cột-Xác suất hình ảnh là gì )

predictions[0]      #Hiển thị kết quả dự đoán xác suất cho hình ảnh đầu tiên trong tập kiểm thử

np.argmax(predictions[0])           #Trả về cái nhãn mà có xác suất cao nhất

test_labels[0]              #Sẽ hiển thị Nhãn thực tế của hình ảnh đầu



#Hiển thị hình ảnh và nhãn
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100*np.max(predictions_array), class_names[true_label]), color=color)

#Biểu đồ xác suất
def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


#Xem kết quả dự đoán
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# Vẽ biểu đồ các hình ảnh kiểm tra X đầu tiên, nhãn dự đoán của chúng và nhãn thực.
# Tô màu dự đoán đúng bằng màu xanh lam và dự đoán sai bằng màu đỏ.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()



# Trả về kích thước của ảnh thứ 2
img = test_images[1]
print(img.shape)

# Thêm chiều để xử lý hình ảnh ( như kiểu từ 2 chiều lên 3 chiều )
img = (np.expand_dims(img,0))
print(img.shape)

#In ra kết quả xác suất với mỗi nhãn
predictions_single = probability_model.predict(img)
print(predictions_single)

#Vẽ biểu đồ xem xác suất với mỗi nhãn
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()


#Chỉ ra nhãn có xác suất cao nhất

np.argmax(predictions_single[0]) 
