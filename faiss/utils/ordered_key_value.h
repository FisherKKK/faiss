/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <climits>
#include <cmath>

#include <limits>

namespace faiss {

/*******************************************************************
 * C object: uniform handling of min and max heap
 *******************************************************************/

/** The C object gives the type T of the values of a key-value storage, the type
 *  of the keys, TI and the comparison that is done: CMax for a decreasing
 *  series and CMin for increasing series. In other words, for a given threshold
 *  threshold, an incoming value x is kept if
 *
 *      C::cmp(threshold, x)
 *
 *  is true.
 */

template <typename T_, typename TI_>
struct CMax;

template <typename T>
inline T cmin_nextafter(T x);
template <typename T>
inline T cmax_nextafter(T x);

// traits of minheaps = heaps where the minimum value is stored on top
// useful to find the *max* values of an array
// 相当于这是小根堆
template <typename T_, typename TI_>
struct CMin {
    typedef T_ T;
    typedef TI_ TI;
    typedef CMax<T_, TI_> Crev; // reference to reverse comparison
    // 实现了cmp函数
    inline static bool cmp(T a, T b) {
        return a < b;
    }
    // Similar to cmp(), but also breaks ties
    // by comparing the second pair of arguments.
    // 比较第二个元素(除了第一个元素之外)
    inline static bool cmp2(T a1, T b1, TI a2, TI b2) {
        return (a1 < b1) || ((a1 == b1) && (a2 < b2));
    }

    // 最小值
    inline static T neutral() {
        return std::numeric_limits<T>::lowest();
    }
    static const bool is_max = false;

    inline static T nextafter(T x) {
        return cmin_nextafter(x);
    }
};

// 相当于T_是实际数据类型, TI_是实际的label的类型
template <typename T_, typename TI_>
struct CMax {
    typedef T_ T;
    typedef TI_ TI;
    typedef CMin<T_, TI_> Crev;
    inline static bool cmp(T a, T b) {
        return a > b;
    }
    // Similar to cmp(), but also breaks ties
    // by comparing the second pair of arguments.
    inline static bool cmp2(T a1, T b1, TI a2, TI b2) {
        return (a1 > b1) || ((a1 == b1) && (a2 > b2));
    }
    inline static T neutral() {
        return std::numeric_limits<T>::max();
    }
    static const bool is_max = true;
    inline static T nextafter(T x) {
        return cmax_nextafter(x);
    }
};

// next_after的模板实现
// 如果采用模板的方式就不会存在多次定义的问题
template <>
inline float cmin_nextafter<float>(float x) {
    return std::nextafterf(x, -HUGE_VALF);
}

template <>
inline float cmax_nextafter<float>(float x) {
    return std::nextafterf(x, HUGE_VALF);
}

template <>
inline uint16_t cmin_nextafter<uint16_t>(uint16_t x) {
    return x - 1;
}

template <>
inline uint16_t cmax_nextafter<uint16_t>(uint16_t x) {
    return x + 1;
}

} // namespace faiss
