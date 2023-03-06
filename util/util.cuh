#pragma once
#include <random>

#define CHECK_CUDA_CALL(CALL) \
	if (CALL != cudaSuccess) { \
		fprintf(stderr, "CUDA CALL failed: %s @ %d, %s\n", \
				cudaGetErrorString(cudaGetLastError()), \
				__LINE__, __FILE__); \
		exit(-1); \
	}

template <typename T, typename L=size_t>
/**
* @brief Simple contiguous array with pointer and length.
*
* T: Type of each element.
* L: Type of the length field.
*/
struct Array {

    /// TODO: Implement Iterator capabilities for convenience.

    /**
    * @brief Points to the first element of the array.
    */
    T* data;

    /**
    * @brief Length of the array.
    */
    L length;

    /**
    * @brief Default constructor.
    */
    inline __host__ __device__ Array() : data(NULL), length(0) {

    }

    /**
    * @brief Initializes the array from pointer and length.
    * 
    * @param first pointer to the first element of the array.
    * @param length the length of the array. Must be 0 if first == NULL.
    */
    inline __host__ __device__ Array(T* data, L length) : data(data), 
            length(length) { 

    }

    template <typename Container>
    /**
    * @brief Initializes the array from a vector.
    * 
    * @param source container (i.e. vector) with source data.
    * @param length the length of the array. Must be 0 if first == NULL.
    */
    inline Array(const Container& source) : data(source.data()), 
            length(source.size()) { 

    }

    /**
     * @brief Get the element at the indicated index.
     * 
     * @param index the index of the element to get.
    */
    inline __host__ __device__ T& operator[](L index) {
        return data[index];
    }
};

template <typename T, typename O=size_t, typename L=size_t>
/**
* @brief Simple contiguous array with offset (not pointer) and length.
*
* It is understood that data in this array is a sub-section/slice of a larger
* array. Given the start of the larger array, you would use the offset and
* index to locate an element in this array. See get(...) below.
*
* T: Type of the array.
* O: Type of the offset field.
* L: Type of the length field.
*/
struct ArrayByOffset {

    /**
    * @brief Offset from some starting location to the first element.
    * 
    * This is measured in number of elements (not bytes).
    */
    O offset;

    /**
    * @brief Length of the array.
    */
    L length;

    /**
    * @brief Converts this to an actual Array (const).
    * 
    * @param start const-pointer to the first element of the larger array that 
    *   this is part of.
    * @return This array in Array format.
    */
    inline __device__ __host__ Array<const T> toArrayConst(const T* start) const {
        return Array<const T, L>(start + offset, length);
    }

    /**
    * @brief Gets an element from the array.
    * 
    * @param start pointer to the first element of the larger array that this
    *   is part of.
    * @param index index of the element in this array.
    * @return reference to the element in this array.
    */
    inline T& get(T* start, O index) const {
        return start[offset + index]; 
    }
};

/**
* @brief Strongly typed boolean to prevent implicit casting.
*
* I tried looking in the standard library and couldn't find one quite like this.
*/
enum StrongBool : bool {
    True = true,
    False = false
};

template <typename T, typename E>
/**
* @brief Short-hand to generate a uniform-distributed integer.
*
* @param minimum minimum possible value.
* @param maximum maximum possible value.
* @param engine a random number generation engine such as std::mt19937.
* @return integer value between minimum and maximum.
*/
inline T uniformInt(T minimum, T maximum, E& engine) {
    return std::uniform_int_distribution<T>(minimum, maximum)(engine);
}

template <typename T, typename E>
/**
* @brief Short-hand to generate a uniform-distributed real number value.
*
* @param engine a random number generation engine such as std::mt19937.
* @return real number value between 0 and 1.
*/
inline T uniformReal(E& engine) {
    return std::uniform_real_distribution<T>()(engine);
}

template <typename T, typename E>
/**
* @brief Short-hand to generate a normal-distributed real number value.
*
* @param sigma the standard deviation.
* @param engine a random number generation engine such as std::mt19937.
* @return real number value normally distributed around 0 with 
*   sigma standard deviation.
*/
inline T normalReal(float sigma, E& engine) {
    return std::normal_distribution<T>(0, sigma)(engine);
}

template <typename T>
/**
* @brief Computes the sign of the given input value.
*
* @param x the number to take the sign of.
* @return -1 if x < 0; 0 if x = 0; 1 if x > 0
*/
__device__ __host__ T sign(T x) { 
	return x > 0 ? 1 : (x<0 ? -1 : 0);
}