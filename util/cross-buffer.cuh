#pragma once
#include "util/util.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <assert.h>

template <typename T>
/**
* @brief A buffer that stores a host and a device side copy of the same data.
*
* Rather than just being pointers to memory regions, this also tracks the shape
* of the data (such as being a 256x256 grid).
*/
class CrossBuffer {
public:

	/**
	* @brief host-side replica.
	*/
	T* hostBuffer;

	/**
	* @brief device-side replica.
	*/
	T* deviceBuffer;

	/**
	* @brief Dimensions of the data.
	*
	* Number of dimensions should match dimCount.
	*/
	unsigned int* dims;

	/**
	* @brief Number of dimensions.
	*/
	unsigned int dimCount;

	/**
	* @brief Indicates if the host buffer is owned by this and thus should be
	* 	automatically destructed when this is destructed.
	*/
	bool ownsHostBuffer;

	/**
	* @brief Indicates if the host buffer is owned by this and thus should be
	* 	automatically destructed when this is destructed.
	*/
	bool ownsDeviceBuffer;

	template <typename... Dims>
	/**
	* Constructs the buffer.
	*
	* @param hostBuffer the host-side buffer. If NULL, then this will 
	* 	automatically be allocated.
	* @param deviceBuffer the device-side buffer. If NULL, then this will 
	* 	automatically be allocated.
	* @param dims the dimensions of the buffer.
	*/
	CrossBuffer(T* hostBuffer, T* deviceBuffer,
			Dims... dims) : CrossBuffer(hostBuffer, deviceBuffer, 
			StrongBool::False, dims...) {

	}

	template <typename... Dims>
	/**
	* Constructs the buffer.
	*
	* @param hostBuffer the host-side buffer. If NULL, then this will 
	* 	automatically be allocated.
	* @param deviceBuffer the device-side buffer. If NULL, then this will 
	* 	automatically be allocated.
	* @param disableHost if true, disables creating a host buffer if one does
	*	not exist already. This allows creating managed device-side-only 
	*	buffers. Without a host side buffer, calls to transferDeviceToHost will
	*	fail.
	* @param dims the dimensions of the buffer.
	*/
	CrossBuffer(T* hostBuffer, T* deviceBuffer,
			StrongBool disableHost=StrongBool::False,
			Dims... dims) :
			ownsHostBuffer(hostBuffer == NULL && !disableHost),
			ownsDeviceBuffer(deviceBuffer == NULL) {
		constexpr unsigned int DimCount = sizeof...(Dims);
		this->dims = new unsigned int[DimCount];
		this->dimCount = DimCount;
		populateDims(0, dims...);
		
		// Allocate host and/or device buffers if not given.
		bool newHost = false;
		bool newDevice = false;
		unsigned int count = countTotalElements();
		if (hostBuffer == NULL && !disableHost) {
			hostBuffer = new T[count];
			newHost = true;
		}
		if (deviceBuffer == NULL) {
			CHECK_CUDA_CALL(cudaMalloc(&deviceBuffer, count * sizeof(T)));
			newDevice = true;
		}
		this->hostBuffer = hostBuffer;
		this->deviceBuffer = deviceBuffer;

		if (newHost && !newDevice) {
			transferDeviceToHost();
		} else if (newDevice && !newHost && !disableHost) {
			transferHostToDevice();
		}
	}

	/**
	* @brief Destructs this buffer.
	*/
	~CrossBuffer() {
		if (ownsHostBuffer) {
			delete[] hostBuffer;
		}
		if (ownsDeviceBuffer) {
			CHECK_CUDA_CALL(cudaFree(deviceBuffer));
		}
		delete[] dims;
	}

	/**
	* @brief Count the total number of elements in each replica of the buffer.
	*
	* @return total number of elements in each replica of the buffer.
	*/
	inline unsigned int countTotalElements() const {
		unsigned int count = 1;
		for (unsigned int i = 0; i < dimCount; i++) {
			count *= this->dims[i];
		}
		return count;
	}

	/**
	* @brief Converts this to a device-side const array.
	* 
	* @return Array taken with elements from the device-side buffer.
	*/
	inline Array<const T> toDeviceArrayConst() const {
		return Array<const T>(deviceBuffer, countTotalElements());
	}

	template <typename... Dims>
	/**
	* @brief Gets the host-side element at the indicated index.
	*
	* @param indices the indices of the element to get.
	* @return reference to the element at the indicated index.
	*/
	inline T& get(Dims... indices) const {
		unsigned int flatIndex = computeFlatIndex(indices...);
		return hostBuffer[flatIndex];
	}

	template <typename S, typename... Dims>
	/**
	* S: Type of source data.
	*
	* @brief Copies elements from the given source into the host-side array,
	*	starting at the indicated element of the host-side array.
	*
	* @param startIndices the indices of the element to start copying into.
	*/
	inline void copyIn(Array<S> source, Dims... startIndices) const {
		unsigned int flatIndex = computeFlatIndex(startIndices...);
		assert(flatIndex + source.length <= countTotalElements());

		for (unsigned int i = 0; i < source.length; i++) {
			hostBuffer[flatIndex + i] = source.data[i];
		}
	}

	/**
	* @brief Overwrite the host-side copy with the device-side copy.
	*/
	inline void transferHostToDevice() {
		CHECK_CUDA_CALL(cudaMemcpy(deviceBuffer, hostBuffer, 
				countTotalElements() * sizeof(T),
				cudaMemcpyKind::cudaMemcpyHostToDevice));
	}

	/**
	* @brief Overwrite the device-side copy with the host-side copy. 
	*/
	inline void transferDeviceToHost() {
		CHECK_CUDA_CALL(cudaMemcpy(hostBuffer, deviceBuffer, 
				countTotalElements() * sizeof(T),
				cudaMemcpyKind::cudaMemcpyDeviceToHost));
	}
	
	/**
	* @brief Clear the device and/or host buffer.
	*
	* @param host whether to clear the host buffer.
	* @param device whether to clear the device buffer.
	* @param clearValue value to memset to.
	*/
	inline void clear(bool host, bool device, int clearValue = 0) {
		if (host) {
			memset(hostBuffer, clearValue, countTotalElements() * sizeof(T));
		}
		if (device) {
			CHECK_CUDA_CALL(cudaMemset(deviceBuffer, clearValue,
					countTotalElements() * sizeof(T)));
		}
	}

private:

	template <typename... Dims>
	/**
	* @brief Helper using varidic templates to populate the list of dimensions.
	*
	* @param index index of the current dimension being processed.
	* @param dim0 first dimension yet to be processed.
	* @param dims remaining dimensions to be processed.  
	*/
	void populateDims(unsigned int index, unsigned int dim0, Dims... dims) {
		this->dims[index] = dim0;
		if constexpr (sizeof...(dims) >= 1) {
			populateDims(index + 1, dims...);
		}
	}

	template <typename... Dims>
	/**
	* @brief Compute flattened index from a list of indices.
	*
	* @param indices indices of the dimensions to be processed.
	* @return flattened index corresponding to the given indices.
	*/
	unsigned int computeFlatIndex(Dims... indices) const {
		assert(sizeof...(indices) == dimCount);
		return computeFlatIndexHelper(0, indices...);
	}

	template <typename... Dims>
	/**
	* @brief Helper using varidic templates to compute the flattened index from
	* 	a list of indices.
	*
	* @param flatIndex accumulated total index so far.
	* @param index0 index on the first dimension of indices still to process.
	* @param indices indices on the rest of the dimensions to be processed.
	* @return reference to the element at the indicated index.
	*/
	unsigned int computeFlatIndexHelper(unsigned int flatIndex, 
			unsigned int index0, Dims... indices) const {
		constexpr unsigned int DimsLeft = sizeof...(indices);

		unsigned int stride = 1;
		if constexpr (DimsLeft > 0) {
			for (unsigned int i = 0; i < DimsLeft; i++) {
				stride *= this->dims[dimCount - 1 - i];
			}
		}
		flatIndex += stride * index0;

		if constexpr (DimsLeft >= 1) {
			return computeFlatIndexHelper(flatIndex, indices...);
		} else {
			return flatIndex;
		}
	}

};