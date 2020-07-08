#ifndef CUDA_ETUDES_VECTOR_H
#define CUDA_ETUDES_VECTOR_H

#include <cstring>

template<typename T>
class vector {
public:
    vector(int m) : M_(m) {
      this->array_ = new T[m];
      cudaMallocManaged(&this->array_, m * sizeof(T));
    }

    vector(int m, T init) : vector(m) {
      for (int i = 0; i < this->M_; ++i) {
        this->array_[i] = init;
      }
    }

    vector(const vector& other) {
      this->M_ = other->M_;
      this->array_ = new T[this->M_];
      cudaMallocManaged(&this->array_, this->M_ * sizeof(T));
      std::memcpy(this->array_, other.array_, this->M_);
    }

    vector& operator=(const vector& other) {
      if (this == &other) return *this;
      if (this->array_ != NULL) cudaFree(this->array_);

      this->M_ = other->M_;
      this->array_ = new T[this->M_];
      cudaMallocManaged(&this->array_, this->M_ * sizeof(T));
      std::memcpy(this->array_, other->array_, this->M_);
      return *this;
    }

    ~vector() {
      cudaFree(this->array_);
    }

    T& operator()(int i)
    {
      return this->array_[i];
    }

private:
    const int M_;
    T* array_;
};

#endif //CUDA_ETUDES_VECTOR_H
