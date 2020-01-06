# using predictive filter flow for face2face warping
I recently found out about an interesting technique called predictive filter flow (pFF), proposed by Shu Kong and Charless Fowlkes (https://arxiv.org/abs/1811.11482).
The key idea is to use Convolutional Neural Networks to learn input-dependent spatially varying filters, i.e. a different filter for each position of the input image. By using the softmax activation on the learned filters, pFF can e.g. be used to implement image warping: Instead of learning offsets for each position we directly learn to apply a weighted sum of the surrounding pixels: Much easier to implement and expand in my opinion.  The original paper covers denoising and image enhancement and a follow up paper discusses computing optical flow from videos (https://arxiv.org/abs/1904.01693).

I thought pFF could make an interesting regularization for learning face swapping, i.e. turning the face of one person into the face of another: Instead of a classical encoder-decoder approach, the network can only move pixels of the input image in order to turn one face into the other. As an example I created a small data set of my face, taken with an old webcam and some images of Jackie Chan, because Jackie Chan is awesome. I used the excellent face_alignment library from Adrian Bulat (https://github.com/1adrianb/face-alignment) to find pairs with similar positioning to create a roughly aligned data set. The results are a bit blurry but translate amazingly well to novel face expressions as shown in the video below.

<p align="center">
  <img width="600" height="480" src=fake.gif>
</p>
