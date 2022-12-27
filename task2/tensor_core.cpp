#include "tensor_core.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <stdlib.h>
#include <iomanip>

using namespace std;
void print_hex(uint32_t num) {
  std::cout << "0x" << std::setw(8) << std::setfill('0') << std::hex << num << std::endl;
}

void print_reg(GPU& volta, unsigned Rid) {
  for (int i = 0; i < volta.total_thread(); ++i) {
    printf("Reg %d T%d: ", Rid, i);
    // print_binary(volta.reg_content(Rid, i));
    print_hex(volta.reg_content(Rid, i));
    // printf(" ");
  }
  printf("\n\n");
} 

void print_reg_val(GPU& volta, unsigned Rid, unsigned bits=16) {
  for (int i = 0; i < volta.total_thread(); ++i) {
    printf("Reg %d T%d: ", Rid, i);
    // print_binary(volta.reg_content(Rid, i));
    unsigned val1 = volta.reg_content(Rid, i);
    unsigned val2 = volta.reg_content(Rid + 1, i);
    uint64_t val = concat(val2, val1);
    if (bits == 16) {
      // print half
      __half* true_val = (__half *)val;
      printf("%.2f\n", __half2float(*true_val));
    } else if (bits == 32) {
      printf("%.4f\n", *(float *)val);
    }
    // printf(" ");
  }
  printf("\n\n");
}

void print_load(GPU& volta, unsigned Rid, unsigned bits=16, unsigned sz=128) {
  for (int i = 0; i < volta.total_thread(); ++i) {
    // print_binary(volta.reg_content(Rid, i));
    printf("T%d ", i);
    if (sz == 128) {
      // unsigned val1 = volta.reg_content(Rid, i);
      // unsigned val2 = volta.reg_content(Rid + 1, i);
      // unsigned val3 = volta.reg_content(Rid + 2, i);
      // unsigned val4 = volta.reg_content(Rid + 3, i);  
      for (int shift = 0; shift < 4; ++shift) {
        printf("Reg %d: ", Rid + shift);
        unsigned val_shift = volta.reg_content(Rid + shift, i);
        __half val_1((unsigned short)(val_shift >> 16));
        __half val_0((unsigned short)(val_shift & 0xffff));
        printf("%.2f %.2f ", __half2float(val_0), __half2float(val_1));
        
      }
      printf("\n");
    }
  else {
    // unsigned val1 = volta.reg_content(Rid, i);
    // unsigned val2 = volta.reg_content(Rid + 1, i);
    for (int shift = 0; shift < 2; ++shift) {
        printf("Reg %d: ", Rid + shift);
        unsigned val_shift = volta.reg_content(Rid + shift, i);
        __half val_1((unsigned short)(val_shift >> 16));
        __half val_0((unsigned short)(val_shift & 0xffff));
        printf("%.2f %.2f ", __half2float(val_0), __half2float(val_1));
        
      }
      printf("\n");
  }
  }
  printf("\n\n");
}

void print_hmma_step0(GPU& volta, unsigned Rid) {
  for (int i = 0; i < volta.total_thread(); ++i) {
    unsigned val_0 = volta.reg_content(Rid, i);
    unsigned val_1 = volta.reg_content(Rid + 1, i);
    printf("T%d Reg %d: %.2f ", i, Rid, *(float *)&val_0);
    printf("T%d Reg %d: %.2f\n", i, Rid + 1, *(float *)&val_1);
  }
  printf("\n");
}

__half __float2half(const float &a) {
  uint32_t bits = *(uint32_t*)&a;
  int sign = (bits >> 31) & 0x1;
  int exponent = (bits >> 23) & 0xff;
  int mantissa = bits & 0x007fffff;
  __half result;
  if (exponent == 0) {
    // Denormalized number --> Underflow
    exponent = mantissa = 0;
  } else if (exponent == 0xff) {
    // Inf or NaN
    exponent = 0x1f;
    mantissa = mantissa ? 0x200 : 0;
  } else {
    // Normalized number
    if (exponent - 127 > 15) {
      // Overflow
      exponent = 0x1f;
      mantissa = 0;
    } else if (exponent - 127 >= -14) {
      // Normalized float -> normalized half, true exp in range [-14, 15], exp in range [113, 142]
      // exp - 127 + 15 = exp - 112
      exponent = exponent - 112;
      mantissa = (mantissa >> 13) & 0x3ff;
    } else if (exponent - 127 >= -24) {
      // Normalized float -> denormalized half, true exp in range [-24, -15], exp in range [103, 112]
      // exp - 103 = exp - 127 + 24, 126 - exp = -exp - 1 + 127
      mantissa = ((1 << (exponent - 103)) + (mantissa >> (126 - exponent))) & 0x3ff;
      exponent = 0;
    } else {
      // Underflow
      exponent = mantissa = 0;
    }
  }
  result.setX((unsigned short)((sign << 15) | (exponent << 10) | mantissa));
  return result;
}

void print_binary(uint32_t bits) {
  char binary[33];
  for (int i = 0; i < 32; ++i) {
    binary[31 - i] = (bits & (1 << i)) ? '1' : '0';
  }
  binary[32] = '\0';
  printf("%s\n", binary);
}

float __half2float(const __half &a) {
  uint16_t bits = (uint16_t)(a.getX());
  int sign = (bits >> 15) & 1;
  int exponent = (bits >> 10) & 0x1f;
  int mantissa = bits & 0x3ff;
  int result;
  if (exponent == 0) {
    // Denormalized number
    // return (float)(pow(-1, sign) * pow(2, -14) * (mantissa / pow(2, 10)));
    int highest_bit = 0;
    int temp = mantissa;
    while (temp) {
      ++highest_bit;
      temp >>= 1;
    }
    // exponent = - 14 - (11 - highest_bit) + 127;
    exponent = 102 + highest_bit;
    mantissa = ((1 << (exponent - 103)) + (mantissa >> (126 - exponent))) & 0x7fffff;
  } else if (exponent == 0x1f) {
    // Inf or NaN
    // return (float)((exponent == 0x1f && mantissa == 0) ? pow(-1, sign) * INFINITY : NAN);
    exponent = 0xff;
    mantissa = mantissa ? 0x400000 : 0x0;
  } else {
    // Normalized number
    // return (float)(pow(-1, sign) * pow(2, exponent - 15) * (1 + mantissa / pow(2, 10)));
    // exp_f - 127 = exp_h - 15, exp_f = exp_h + 112
    exponent = exponent + 112;
    mantissa = mantissa << 13;
  }
  result = (sign << 31) | (exponent << 23) | mantissa;
  return *reinterpret_cast<float*>(&result);
}

float operator*(const __half &lh, const __half &rh) {
  float a = __half2float(lh);
  float b = __half2float(rh);
  return a * b;
}

GPU::GPU() {
  // Initialize GPU resources reasonably, including regfile size and global
  // memory size, assuming sm=1 and warp_num=1
  constexpr unsigned MEMORY_SIZE = 1024 * 1024 * 1024;  // 1 GB
  memory_ = new unsigned[MEMORY_SIZE];
  memory_size_ = (uint64_t) (sizeof(unsigned) * MEMORY_SIZE);
  // Initialize the register file and pre-register file
  unsigned REGFILE_SIZE = WARP_SIZE_ * 256;
  regfile_ = new unsigned[REGFILE_SIZE];
  for (int i = 0; i < WARP_SIZE_; ++i) {
    for (int j = 0; j <=255; ++j){
      regfile_[j * WARP_SIZE_ + i] = 0;

    }
  }
  unsigned PREGFILE_SIZE = WARP_SIZE_ * 8;
  pregfile_ = new bool[PREGFILE_SIZE];
  for (int i = 0; i < WARP_SIZE_; ++i) {
    pregfile_[7 * WARP_SIZE_ + i] = true;
  }
  allocated_size_ = 0;
}

uint64_t concat(unsigned high, unsigned low) {
  uint64_t result = high;
  result = (result << 32) | low;
  return result;
}

void split(uint64_t data, unsigned &high, unsigned &low) {
  high = data >> 32;
  low = data & 0xffffffff;
}

/**
 * @brief Load data from global memory to register file
 * @param E: 1 if the address is 64-bit (load from Ra - Ra+1), 0 if the address is 32-bit (load from Ra)
 * @param sz: 64 means load 64-bit data, 128 means load 128-bit data
 * @param Rd: the first destination register (sz = 64 means load to Rd to Rd + 1 registers, sz = 128 means load to Rd to Rd + 3 registers)
 * @param Ra: the first source register containing the base address
 * @param imm: the immediate value that contains the offset
 */
void GPU::SIM_LDG_INSTR(bool E, unsigned sz, unsigned Rd, unsigned Ra, unsigned imm) {
  // for: warp execution
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
    // LDG implementation
    unsigned *data_ptr;
    if (E) {
      unsigned &ra_data_0 = regfile_[Ra * WARP_SIZE_ + threadIdx];
      unsigned &ra_data_1 = regfile_[(Ra + 1) * WARP_SIZE_ + threadIdx];
      uint64_t addr = concat(ra_data_1, ra_data_0) + (uint64_t)imm;
      data_ptr = (unsigned *)addr;
    } else {
      unsigned &ra_data = regfile_[Ra * WARP_SIZE_ + threadIdx];
      unsigned addr = ra_data + imm;
      data_ptr = (unsigned *)addr;
    }
    switch (sz) {
      case 64:
        // 64-bit data
        regfile_[Rd * WARP_SIZE_ + threadIdx] = *data_ptr;
        regfile_[(Rd + 1) * WARP_SIZE_ + threadIdx] = *(data_ptr + 1);
        break;
      case 128:
        // 128-bit data
        regfile_[Rd * WARP_SIZE_ + threadIdx] = *data_ptr;
        regfile_[(Rd + 1) * WARP_SIZE_ + threadIdx] = *(data_ptr + 1);
        regfile_[(Rd + 2) * WARP_SIZE_ + threadIdx] = *(data_ptr + 2);
        regfile_[(Rd + 3) * WARP_SIZE_ + threadIdx] = *(data_ptr + 3);
        break;
    }
  }
}

/**
 * @brief Store data from register file to global memory
 * @param E: 1 if the address is 64-bit (load from Ra - Ra+1), 0 if the address is 32-bit (load from Ra)
 * @param sz: 64 means load 64-bit data, 128 means load 128-bit data
 * @param Sb: the first source register (sz = 64 means store data from Rd to Rd + 1 registers, sz = 128 means store data from Rd to Rd + 3 registers)
 * @param Ra: the first source register containing the base address
 * @param imm: the immediate value that contains the offset
 */
void GPU::SIM_STG_INSTR(bool E, unsigned sz, unsigned Sb, unsigned Ra, unsigned imm) {
  // STG implementation
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; ++threadIdx) {
    unsigned * data_ptr;
    if (E) {
      unsigned &ra_data_0 = regfile_[Ra * WARP_SIZE_ + threadIdx];
      unsigned &ra_data_1 = regfile_[(Ra + 1) * WARP_SIZE_ + threadIdx];
      uint64_t addr = concat(ra_data_1, ra_data_0) + (uint64_t)imm;
      data_ptr = (unsigned *)addr;
    } else {
      unsigned &ra_data = regfile_[Ra * WARP_SIZE_ + threadIdx];
      unsigned addr = ra_data + imm;
      data_ptr = (unsigned *)addr;
    }
    switch (sz) {
      case 64:
        *(data_ptr) = regfile_[Sb * WARP_SIZE_ + threadIdx];
        *(data_ptr + 1) = regfile_[(Sb + 1) * WARP_SIZE_ + threadIdx];
        break;
      case 128:
        *(data_ptr) = regfile_[Sb * WARP_SIZE_ + threadIdx];
        *(data_ptr + 1) = regfile_[(Sb + 1) * WARP_SIZE_ + threadIdx];
        *(data_ptr + 2) = regfile_[(Sb + 2) * WARP_SIZE_ + threadIdx];
        *(data_ptr + 3) = regfile_[(Sb + 3) * WARP_SIZE_ + threadIdx];
        break;
    }
  }
}

__half parse_half(unsigned origin, bool first) {
  if (first) {
    return __half(origin & 0xffff);
  } else {
    return __half(origin >> 16);
  }
}

/** HMMA simulation instructions for step 0
 * @param a_fmt 1 for row major, 0 for col major. Default to 1
 * @param b_fmt 1 for row major, 0 for col major. Default to 1
 * @param Rd result stores to Rd and Rd + 1 register
 * @param Ra matrix a stored in Ra and Ra + 1 register
 * @param Sb matrix b stored in Sb and Sb + 1 register
 * @param Sc matric c stored in Sc and Sc + 1 register
*/
void GPU::SIM_HMMA_INSTR_STEP0(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc, unsigned a_fmt, unsigned b_fmt) {
  if (a_fmt && b_fmt) {
    // only implement row major
    for (int octet_idx = 0; octet_idx < 4; ++octet_idx) {
      int thread_group_idx_0 = octet_idx;
      int thread_group_idx_1 = octet_idx + 4;

      // first thread group
      for (int thread_matrix_a_idx = 4 * thread_group_idx_0; thread_matrix_a_idx < 4 * thread_group_idx_0 + 2; ++thread_matrix_a_idx) {
        __half a_data_0 = parse_half(regfile_[Ra * WARP_SIZE_ + thread_matrix_a_idx], true);
        __half a_data_1 = parse_half(regfile_[Ra * WARP_SIZE_ + thread_matrix_a_idx], false);
        __half a_data_2 = parse_half(regfile_[(Ra + 1) * WARP_SIZE_ + thread_matrix_a_idx], true);
        __half a_data_3 = parse_half(regfile_[(Ra + 1) * WARP_SIZE_ + thread_matrix_a_idx], false);
        float c_data_0 = *(float*)(&regfile_[Sc * WARP_SIZE_ + thread_matrix_a_idx]);
        float c_data_1 = *(float*)(&regfile_[(Sc + 1) * WARP_SIZE_ + thread_matrix_a_idx]);
        float c_data_2 = *(float*)(&regfile_[Sc * WARP_SIZE_ + thread_matrix_a_idx + 2]);
        float c_data_3 = *(float*)(&regfile_[(Sc + 1) * WARP_SIZE_ + thread_matrix_a_idx + 2]);
        for (int idx = 0; idx < 4; ++idx) {
          int thread_matrix_b_idx = 4 * thread_group_idx_0 + idx;
          __half b_data_0 = parse_half(regfile_[Sb * WARP_SIZE_ + thread_matrix_b_idx], true);
          __half b_data_1 = parse_half(regfile_[Sb * WARP_SIZE_ + thread_matrix_b_idx], false);
          __half b_data_2 = parse_half(regfile_[(Sb + 1) * WARP_SIZE_ + thread_matrix_b_idx], true);
          __half b_data_3 = parse_half(regfile_[(Sb + 1) * WARP_SIZE_ + thread_matrix_b_idx], false);
          if (idx == 0) {
            c_data_0 += a_data_0 * b_data_0;
            c_data_1 += a_data_0 * b_data_1;
            c_data_2 += a_data_0 * b_data_2;
            c_data_3 += a_data_0 * b_data_3;
          } else if (idx == 1) {
            c_data_0 += a_data_1 * b_data_0;
            c_data_1 += a_data_1 * b_data_1;
            c_data_2 += a_data_1 * b_data_2;
            c_data_3 += a_data_1 * b_data_3;
          } else if (idx == 2) {
            c_data_0 += a_data_2 * b_data_0;
            c_data_1 += a_data_2 * b_data_1;
            c_data_2 += a_data_2 * b_data_2;
            c_data_3 += a_data_2 * b_data_3;
          } else if (idx == 3) {
            c_data_0 += a_data_3 * b_data_0;
            c_data_1 += a_data_3 * b_data_1;
            c_data_2 += a_data_3 * b_data_2;
            c_data_3 += a_data_3 * b_data_3;
          }
        }
        regfile_[Rd * WARP_SIZE_ + thread_matrix_a_idx] = *reinterpret_cast<unsigned*>(&c_data_0);
        regfile_[(Rd + 1) * WARP_SIZE_ + thread_matrix_a_idx] = *reinterpret_cast<unsigned*>(&c_data_1);
        regfile_[Rd * WARP_SIZE_ + (thread_matrix_a_idx + 2)] = *reinterpret_cast<unsigned*>(&c_data_2);
        regfile_[(Rd + 1) * WARP_SIZE_ + (thread_matrix_a_idx + 2)] = *reinterpret_cast<unsigned*>(&c_data_3);
      }

      // second thread group
      for (int thread_matrix_a_idx = 4 * thread_group_idx_1; thread_matrix_a_idx < 4 * thread_group_idx_1 + 2; ++thread_matrix_a_idx) {
        __half a_data_0 = parse_half(regfile_[Ra * WARP_SIZE_ + thread_matrix_a_idx], true);
        __half a_data_1 = parse_half(regfile_[Ra * WARP_SIZE_ + thread_matrix_a_idx], false);
        __half a_data_2 = parse_half(regfile_[(Ra + 1) * WARP_SIZE_ + thread_matrix_a_idx], true);
        __half a_data_3 = parse_half(regfile_[(Ra + 1) * WARP_SIZE_ + thread_matrix_a_idx], false);
        float c_data_0 = *(float*)(&regfile_[Sc * WARP_SIZE_ + thread_matrix_a_idx]);
        float c_data_1 = *(float*)(&regfile_[(Sc + 1) * WARP_SIZE_ + thread_matrix_a_idx]);
        float c_data_2 = *(float*)(&regfile_[Sc * WARP_SIZE_ + thread_matrix_a_idx + 2]);
        float c_data_3 = *(float*)(&regfile_[(Sc + 1) * WARP_SIZE_ + thread_matrix_a_idx + 2]);
        for (int idx = 0; idx < 4; ++idx) {
          int thread_matrix_b_idx = 4 * thread_group_idx_0 + idx;
          __half b_data_0 = parse_half(regfile_[Sb * WARP_SIZE_ + thread_matrix_b_idx], true);
          __half b_data_1 = parse_half(regfile_[Sb * WARP_SIZE_ + thread_matrix_b_idx], false);
          __half b_data_2 = parse_half(regfile_[(Sb + 1) * WARP_SIZE_ + thread_matrix_b_idx], true);
          __half b_data_3 = parse_half(regfile_[(Sb + 1) * WARP_SIZE_ + thread_matrix_b_idx], false);

          if (idx == 0) {
            c_data_0 += a_data_0 * b_data_0;
            c_data_1 += a_data_0 * b_data_1;
            c_data_2 += a_data_0 * b_data_2;
            c_data_3 += a_data_0 * b_data_3;
          } else if (idx == 1) {
            c_data_0 += a_data_1 * b_data_0;
            c_data_1 += a_data_1 * b_data_1;
            c_data_2 += a_data_1 * b_data_2;
            c_data_3 += a_data_1 * b_data_3;
          } else if (idx == 2) {
            c_data_0 += a_data_2 * b_data_0;
            c_data_1 += a_data_2 * b_data_1;
            c_data_2 += a_data_2 * b_data_2;
            c_data_3 += a_data_2 * b_data_3;
          } else if (idx == 3) {
            c_data_0 += a_data_3 * b_data_0;
            c_data_1 += a_data_3 * b_data_1;
            c_data_2 += a_data_3 * b_data_2;
            c_data_3 += a_data_3 * b_data_3;
          }
        }
        regfile_[Rd * WARP_SIZE_ + thread_matrix_a_idx] = *reinterpret_cast<unsigned*>(&c_data_0);
        regfile_[(Rd + 1) * WARP_SIZE_ + thread_matrix_a_idx] = *reinterpret_cast<unsigned*>(&c_data_1);
        regfile_[Rd * WARP_SIZE_ + (thread_matrix_a_idx + 2)] = *reinterpret_cast<unsigned*>(&c_data_2);
        regfile_[(Rd + 1) * WARP_SIZE_ + (thread_matrix_a_idx + 2)] = *reinterpret_cast<unsigned*>(&c_data_3);
      }
    }
  }
}
void GPU::SIM_HMMA_INSTR_STEP1(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc, unsigned a_fmt, unsigned b_fmt) {
  if (a_fmt && b_fmt) {
    // only implement row major
    for (int octet_idx = 0; octet_idx < 4; ++octet_idx) {
      int thread_group_idx_0 = octet_idx;
      int thread_group_idx_1 = octet_idx + 4;

      // first thread group
      for (int thread_matrix_a_idx = 4 * thread_group_idx_0 + 2; thread_matrix_a_idx < 4 * thread_group_idx_0 + 4; ++thread_matrix_a_idx) {
        __half a_data_0 = parse_half(regfile_[Ra * WARP_SIZE_ + thread_matrix_a_idx], true);
        __half a_data_1 = parse_half(regfile_[Ra * WARP_SIZE_ + thread_matrix_a_idx], false);
        __half a_data_2 = parse_half(regfile_[(Ra + 1) * WARP_SIZE_ + thread_matrix_a_idx], true);
        __half a_data_3 = parse_half(regfile_[(Ra + 1) * WARP_SIZE_ + thread_matrix_a_idx], false);
        float c_data_0 = *(float*)(&regfile_[Sc * WARP_SIZE_ + (thread_matrix_a_idx - 2)]);
        float c_data_1 = *(float*)(&regfile_[(Sc + 1) * WARP_SIZE_ + (thread_matrix_a_idx - 2)]);
        float c_data_2 = *(float*)(&regfile_[Sc * WARP_SIZE_ + thread_matrix_a_idx]);
        float c_data_3 = *(float*)(&regfile_[(Sc + 1) * WARP_SIZE_ + thread_matrix_a_idx]);
        for (int idx = 0; idx < 4; ++idx) {
          int thread_matrix_b_idx = 4 * thread_group_idx_0 + idx;
          __half b_data_0 = parse_half(regfile_[Sb * WARP_SIZE_ + thread_matrix_b_idx], true);
          __half b_data_1 = parse_half(regfile_[Sb * WARP_SIZE_ + thread_matrix_b_idx], false);
          __half b_data_2 = parse_half(regfile_[(Sb + 1) * WARP_SIZE_ + thread_matrix_b_idx], true);
          __half b_data_3 = parse_half(regfile_[(Sb + 1) * WARP_SIZE_ + thread_matrix_b_idx], false);
          if (idx == 0) {
            c_data_0 += a_data_0 * b_data_0;
            c_data_1 += a_data_0 * b_data_1;
            c_data_2 += a_data_0 * b_data_2;
            c_data_3 += a_data_0 * b_data_3;
          } else if (idx == 1) {
            c_data_0 += a_data_1 * b_data_0;
            c_data_1 += a_data_1 * b_data_1;
            c_data_2 += a_data_1 * b_data_2;
            c_data_3 += a_data_1 * b_data_3;
          } else if (idx == 2) {
            c_data_0 += a_data_2 * b_data_0;
            c_data_1 += a_data_2 * b_data_1;
            c_data_2 += a_data_2 * b_data_2;
            c_data_3 += a_data_2 * b_data_3;
          } else if (idx == 3) {
            c_data_0 += a_data_3 * b_data_0;
            c_data_1 += a_data_3 * b_data_1;
            c_data_2 += a_data_3 * b_data_2;
            c_data_3 += a_data_3 * b_data_3;
          }
        }
        regfile_[Rd * WARP_SIZE_ + (thread_matrix_a_idx - 2)] = *reinterpret_cast<unsigned*>(&c_data_0);
        regfile_[(Rd + 1) * WARP_SIZE_ + (thread_matrix_a_idx - 2)] = *reinterpret_cast<unsigned*>(&c_data_1);
        regfile_[Rd * WARP_SIZE_ + thread_matrix_a_idx] = *reinterpret_cast<unsigned*>(&c_data_2);
        regfile_[(Rd + 1) * WARP_SIZE_ + thread_matrix_a_idx] = *reinterpret_cast<unsigned*>(&c_data_3);
      }

      // second thread group
      for (int thread_matrix_a_idx = 4 * thread_group_idx_1 + 2; thread_matrix_a_idx < 4 * thread_group_idx_1 + 4; ++thread_matrix_a_idx) {
        __half a_data_0 = parse_half(regfile_[Ra * WARP_SIZE_ + thread_matrix_a_idx], true);
        __half a_data_1 = parse_half(regfile_[Ra * WARP_SIZE_ + thread_matrix_a_idx], false);
        __half a_data_2 = parse_half(regfile_[(Ra + 1) * WARP_SIZE_ + thread_matrix_a_idx], true);
        __half a_data_3 = parse_half(regfile_[(Ra + 1) * WARP_SIZE_ + thread_matrix_a_idx], false);
        float c_data_0 = *(float*)(&regfile_[Sc * WARP_SIZE_ + (thread_matrix_a_idx - 2)]);
        float c_data_1 = *(float*)(&regfile_[(Sc + 1) * WARP_SIZE_ + (thread_matrix_a_idx - 2)]);
        float c_data_2 = *(float*)(&regfile_[Sc * WARP_SIZE_ + thread_matrix_a_idx]);
        float c_data_3 = *(float*)(&regfile_[(Sc + 1) * WARP_SIZE_ + thread_matrix_a_idx]);
        for (int idx = 0; idx < 4; ++idx) {
          int thread_matrix_b_idx = 4 * thread_group_idx_0 + idx;
          __half b_data_0 = parse_half(regfile_[Sb * WARP_SIZE_ + thread_matrix_b_idx], true);
          __half b_data_1 = parse_half(regfile_[Sb * WARP_SIZE_ + thread_matrix_b_idx], false);
          __half b_data_2 = parse_half(regfile_[(Sb + 1) * WARP_SIZE_ + thread_matrix_b_idx], true);
          __half b_data_3 = parse_half(regfile_[(Sb + 1) * WARP_SIZE_ + thread_matrix_b_idx], false);

          if (idx == 0) {
            c_data_0 += a_data_0 * b_data_0;
            c_data_1 += a_data_0 * b_data_1;
            c_data_2 += a_data_0 * b_data_2;
            c_data_3 += a_data_0 * b_data_3;
          } else if (idx == 1) {
            c_data_0 += a_data_1 * b_data_0;
            c_data_1 += a_data_1 * b_data_1;
            c_data_2 += a_data_1 * b_data_2;
            c_data_3 += a_data_1 * b_data_3;
          } else if (idx == 2) {
            c_data_0 += a_data_2 * b_data_0;
            c_data_1 += a_data_2 * b_data_1;
            c_data_2 += a_data_2 * b_data_2;
            c_data_3 += a_data_2 * b_data_3;
          } else if (idx == 3) {
            c_data_0 += a_data_3 * b_data_0;
            c_data_1 += a_data_3 * b_data_1;
            c_data_2 += a_data_3 * b_data_2;
            c_data_3 += a_data_3 * b_data_3;
          }
        }
        regfile_[Rd * WARP_SIZE_ + (thread_matrix_a_idx - 2)] = *reinterpret_cast<unsigned*>(&c_data_0);
        regfile_[(Rd + 1) * WARP_SIZE_ + (thread_matrix_a_idx - 2)] = *reinterpret_cast<unsigned*>(&c_data_1);
        regfile_[Rd * WARP_SIZE_ + thread_matrix_a_idx] = *reinterpret_cast<unsigned*>(&c_data_2);
        regfile_[(Rd + 1) * WARP_SIZE_ + thread_matrix_a_idx] = *reinterpret_cast<unsigned*>(&c_data_3);
      }
    }
  }
}
void GPU::SIM_HMMA_INSTR_STEP2(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc, unsigned a_fmt, unsigned b_fmt) {
  if (a_fmt && b_fmt) {
    // only implement row major
    for (int octet_idx = 0; octet_idx < 4; ++octet_idx) {
      int thread_group_idx_0 = octet_idx;
      int thread_group_idx_1 = octet_idx + 4;

      // first thread group
      for (int thread_matrix_a_idx = 4 * thread_group_idx_0; thread_matrix_a_idx < 4 * thread_group_idx_0 + 2; ++thread_matrix_a_idx) {
        __half a_data_0 = parse_half(regfile_[Ra * WARP_SIZE_ + thread_matrix_a_idx], true);
        __half a_data_1 = parse_half(regfile_[Ra * WARP_SIZE_ + thread_matrix_a_idx], false);
        __half a_data_2 = parse_half(regfile_[(Ra + 1) * WARP_SIZE_ + thread_matrix_a_idx], true);
        __half a_data_3 = parse_half(regfile_[(Ra + 1) * WARP_SIZE_ + thread_matrix_a_idx], false);
        float c_data_0 = *(float*)(&regfile_[Sc * WARP_SIZE_ + thread_matrix_a_idx]);
        float c_data_1 = *(float*)(&regfile_[(Sc + 1) * WARP_SIZE_ + thread_matrix_a_idx]);
        float c_data_2 = *(float*)(&regfile_[Sc * WARP_SIZE_ + thread_matrix_a_idx + 2]);
        float c_data_3 = *(float*)(&regfile_[(Sc + 1) * WARP_SIZE_ + thread_matrix_a_idx + 2]);
        for (int idx = 0; idx < 4; ++idx) {
          int thread_matrix_b_idx = 4 * thread_group_idx_1 + idx;
          __half b_data_0 = parse_half(regfile_[Sb * WARP_SIZE_ + thread_matrix_b_idx], true);
          __half b_data_1 = parse_half(regfile_[Sb * WARP_SIZE_ + thread_matrix_b_idx], false);
          __half b_data_2 = parse_half(regfile_[(Sb + 1) * WARP_SIZE_ + thread_matrix_b_idx], true);
          __half b_data_3 = parse_half(regfile_[(Sb + 1) * WARP_SIZE_ + thread_matrix_b_idx], false);
          if (idx == 0) {
            c_data_0 += a_data_0 * b_data_0;
            c_data_1 += a_data_0 * b_data_1;
            c_data_2 += a_data_0 * b_data_2;
            c_data_3 += a_data_0 * b_data_3;
          } else if (idx == 1) {
            c_data_0 += a_data_1 * b_data_0;
            c_data_1 += a_data_1 * b_data_1;
            c_data_2 += a_data_1 * b_data_2;
            c_data_3 += a_data_1 * b_data_3;
          } else if (idx == 2) {
            c_data_0 += a_data_2 * b_data_0;
            c_data_1 += a_data_2 * b_data_1;
            c_data_2 += a_data_2 * b_data_2;
            c_data_3 += a_data_2 * b_data_3;
          } else if (idx == 3) {
            c_data_0 += a_data_3 * b_data_0;
            c_data_1 += a_data_3 * b_data_1;
            c_data_2 += a_data_3 * b_data_2;
            c_data_3 += a_data_3 * b_data_3;
          }
        }
        regfile_[Rd * WARP_SIZE_ + thread_matrix_a_idx] = *reinterpret_cast<unsigned*>(&c_data_0);
        regfile_[(Rd + 1) * WARP_SIZE_ + thread_matrix_a_idx] = *reinterpret_cast<unsigned*>(&c_data_1);
        regfile_[Rd * WARP_SIZE_ + (thread_matrix_a_idx + 2)] = *reinterpret_cast<unsigned*>(&c_data_2);
        regfile_[(Rd + 1) * WARP_SIZE_ + (thread_matrix_a_idx + 2)] = *reinterpret_cast<unsigned*>(&c_data_3);
      }

      // second thread group
      for (int thread_matrix_a_idx = 4 * thread_group_idx_1; thread_matrix_a_idx < 4 * thread_group_idx_1 + 2; ++thread_matrix_a_idx) {
        __half a_data_0 = parse_half(regfile_[Ra * WARP_SIZE_ + thread_matrix_a_idx], true);
        __half a_data_1 = parse_half(regfile_[Ra * WARP_SIZE_ + thread_matrix_a_idx], false);
        __half a_data_2 = parse_half(regfile_[(Ra + 1) * WARP_SIZE_ + thread_matrix_a_idx], true);
        __half a_data_3 = parse_half(regfile_[(Ra + 1) * WARP_SIZE_ + thread_matrix_a_idx], false);
        float c_data_0 = *(float*)(&regfile_[Sc * WARP_SIZE_ + thread_matrix_a_idx]);
        float c_data_1 = *(float*)(&regfile_[(Sc + 1) * WARP_SIZE_ + thread_matrix_a_idx]);
        float c_data_2 = *(float*)(&regfile_[Sc * WARP_SIZE_ + thread_matrix_a_idx + 2]);
        float c_data_3 = *(float*)(&regfile_[(Sc + 1) * WARP_SIZE_ + thread_matrix_a_idx + 2]);
        for (int idx = 0; idx < 4; ++idx) {
          int thread_matrix_b_idx = 4 * thread_group_idx_1 + idx;
          __half b_data_0 = parse_half(regfile_[Sb * WARP_SIZE_ + thread_matrix_b_idx], true);
          __half b_data_1 = parse_half(regfile_[Sb * WARP_SIZE_ + thread_matrix_b_idx], false);
          __half b_data_2 = parse_half(regfile_[(Sb + 1) * WARP_SIZE_ + thread_matrix_b_idx], true);
          __half b_data_3 = parse_half(regfile_[(Sb + 1) * WARP_SIZE_ + thread_matrix_b_idx], false);

          if (idx == 0) {
            c_data_0 += a_data_0 * b_data_0;
            c_data_1 += a_data_0 * b_data_1;
            c_data_2 += a_data_0 * b_data_2;
            c_data_3 += a_data_0 * b_data_3;
          } else if (idx == 1) {
            c_data_0 += a_data_1 * b_data_0;
            c_data_1 += a_data_1 * b_data_1;
            c_data_2 += a_data_1 * b_data_2;
            c_data_3 += a_data_1 * b_data_3;
          } else if (idx == 2) {
            c_data_0 += a_data_2 * b_data_0;
            c_data_1 += a_data_2 * b_data_1;
            c_data_2 += a_data_2 * b_data_2;
            c_data_3 += a_data_2 * b_data_3;
          } else if (idx == 3) {
            c_data_0 += a_data_3 * b_data_0;
            c_data_1 += a_data_3 * b_data_1;
            c_data_2 += a_data_3 * b_data_2;
            c_data_3 += a_data_3 * b_data_3;
          }
        }
        regfile_[Rd * WARP_SIZE_ + thread_matrix_a_idx] = *reinterpret_cast<unsigned*>(&c_data_0);
        regfile_[(Rd + 1) * WARP_SIZE_ + thread_matrix_a_idx] = *reinterpret_cast<unsigned*>(&c_data_1);
        regfile_[Rd * WARP_SIZE_ + (thread_matrix_a_idx + 2)] = *reinterpret_cast<unsigned*>(&c_data_2);
        regfile_[(Rd + 1) * WARP_SIZE_ + (thread_matrix_a_idx + 2)] = *reinterpret_cast<unsigned*>(&c_data_3);
      }
    }
  }
}
void GPU::SIM_HMMA_INSTR_STEP3(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc, unsigned a_fmt, unsigned b_fmt) {
  if (a_fmt && b_fmt) {
    // only implement row major
    for (int octet_idx = 0; octet_idx < 4; ++octet_idx) {
      int thread_group_idx_0 = octet_idx;
      int thread_group_idx_1 = octet_idx + 4;

      // first thread group
      for (int thread_matrix_a_idx = 4 * thread_group_idx_0 + 2; thread_matrix_a_idx < 4 * thread_group_idx_0 + 4; ++thread_matrix_a_idx) {
        __half a_data_0 = parse_half(regfile_[Ra * WARP_SIZE_ + thread_matrix_a_idx], true);
        __half a_data_1 = parse_half(regfile_[Ra * WARP_SIZE_ + thread_matrix_a_idx], false);
        __half a_data_2 = parse_half(regfile_[(Ra + 1) * WARP_SIZE_ + thread_matrix_a_idx], true);
        __half a_data_3 = parse_half(regfile_[(Ra + 1) * WARP_SIZE_ + thread_matrix_a_idx], false);
        float c_data_0 = *(float*)(&regfile_[Sc * WARP_SIZE_ + (thread_matrix_a_idx - 2)]);
        float c_data_1 = *(float*)(&regfile_[(Sc + 1) * WARP_SIZE_ + (thread_matrix_a_idx - 2)]);
        float c_data_2 = *(float*)(&regfile_[Sc * WARP_SIZE_ + thread_matrix_a_idx]);
        float c_data_3 = *(float*)(&regfile_[(Sc + 1) * WARP_SIZE_ + thread_matrix_a_idx]);
        for (int idx = 0; idx < 4; ++idx) {
          int thread_matrix_b_idx = 4 * thread_group_idx_1 + idx;
          __half b_data_0 = parse_half(regfile_[Sb * WARP_SIZE_ + thread_matrix_b_idx], true);
          __half b_data_1 = parse_half(regfile_[Sb * WARP_SIZE_ + thread_matrix_b_idx], false);
          __half b_data_2 = parse_half(regfile_[(Sb + 1) * WARP_SIZE_ + thread_matrix_b_idx], true);
          __half b_data_3 = parse_half(regfile_[(Sb + 1) * WARP_SIZE_ + thread_matrix_b_idx], false);
          if (idx == 0) {
            c_data_0 += a_data_0 * b_data_0;
            c_data_1 += a_data_0 * b_data_1;
            c_data_2 += a_data_0 * b_data_2;
            c_data_3 += a_data_0 * b_data_3;
          } else if (idx == 1) {
            c_data_0 += a_data_1 * b_data_0;
            c_data_1 += a_data_1 * b_data_1;
            c_data_2 += a_data_1 * b_data_2;
            c_data_3 += a_data_1 * b_data_3;
          } else if (idx == 2) {
            c_data_0 += a_data_2 * b_data_0;
            c_data_1 += a_data_2 * b_data_1;
            c_data_2 += a_data_2 * b_data_2;
            c_data_3 += a_data_2 * b_data_3;
          } else if (idx == 3) {
            c_data_0 += a_data_3 * b_data_0;
            c_data_1 += a_data_3 * b_data_1;
            c_data_2 += a_data_3 * b_data_2;
            c_data_3 += a_data_3 * b_data_3;
          }
        }
        regfile_[Rd * WARP_SIZE_ + (thread_matrix_a_idx - 2)] = *reinterpret_cast<unsigned*>(&c_data_0);
        regfile_[(Rd + 1) * WARP_SIZE_ + (thread_matrix_a_idx - 2)] = *reinterpret_cast<unsigned*>(&c_data_1);
        regfile_[Rd * WARP_SIZE_ + thread_matrix_a_idx] = *reinterpret_cast<unsigned*>(&c_data_2);
        regfile_[(Rd + 1) * WARP_SIZE_ + thread_matrix_a_idx] = *reinterpret_cast<unsigned*>(&c_data_3);
      }

      // second thread group
      for (int thread_matrix_a_idx = 4 * thread_group_idx_1 + 2; thread_matrix_a_idx < 4 * thread_group_idx_1 + 4; ++thread_matrix_a_idx) {
        __half a_data_0 = parse_half(regfile_[Ra * WARP_SIZE_ + thread_matrix_a_idx], true);
        __half a_data_1 = parse_half(regfile_[Ra * WARP_SIZE_ + thread_matrix_a_idx], false);
        __half a_data_2 = parse_half(regfile_[(Ra + 1) * WARP_SIZE_ + thread_matrix_a_idx], true);
        __half a_data_3 = parse_half(regfile_[(Ra + 1) * WARP_SIZE_ + thread_matrix_a_idx], false);
        float c_data_0 = *(float*)(&regfile_[Sc * WARP_SIZE_ + (thread_matrix_a_idx - 2)]);
        float c_data_1 = *(float*)(&regfile_[(Sc + 1) * WARP_SIZE_ + (thread_matrix_a_idx - 2)]);
        float c_data_2 = *(float*)(&regfile_[Sc * WARP_SIZE_ + thread_matrix_a_idx]);
        float c_data_3 = *(float*)(&regfile_[(Sc + 1) * WARP_SIZE_ + thread_matrix_a_idx]);
        for (int idx = 0; idx < 4; ++idx) {
          int thread_matrix_b_idx = 4 * thread_group_idx_1 + idx;
          __half b_data_0 = parse_half(regfile_[Sb * WARP_SIZE_ + thread_matrix_b_idx], true);
          __half b_data_1 = parse_half(regfile_[Sb * WARP_SIZE_ + thread_matrix_b_idx], false);
          __half b_data_2 = parse_half(regfile_[(Sb + 1) * WARP_SIZE_ + thread_matrix_b_idx], true);
          __half b_data_3 = parse_half(regfile_[(Sb + 1) * WARP_SIZE_ + thread_matrix_b_idx], false);

          if (idx == 0) {
            c_data_0 += a_data_0 * b_data_0;
            c_data_1 += a_data_0 * b_data_1;
            c_data_2 += a_data_0 * b_data_2;
            c_data_3 += a_data_0 * b_data_3;
          } else if (idx == 1) {
            c_data_0 += a_data_1 * b_data_0;
            c_data_1 += a_data_1 * b_data_1;
            c_data_2 += a_data_1 * b_data_2;
            c_data_3 += a_data_1 * b_data_3;
          } else if (idx == 2) {
            c_data_0 += a_data_2 * b_data_0;
            c_data_1 += a_data_2 * b_data_1;
            c_data_2 += a_data_2 * b_data_2;
            c_data_3 += a_data_2 * b_data_3;
          } else if (idx == 3) {
            c_data_0 += a_data_3 * b_data_0;
            c_data_1 += a_data_3 * b_data_1;
            c_data_2 += a_data_3 * b_data_2;
            c_data_3 += a_data_3 * b_data_3;
          }
        }
        regfile_[Rd * WARP_SIZE_ + (thread_matrix_a_idx - 2)] = *reinterpret_cast<unsigned*>(&c_data_0);
        regfile_[(Rd + 1) * WARP_SIZE_ + (thread_matrix_a_idx - 2)] = *reinterpret_cast<unsigned*>(&c_data_1);
        regfile_[Rd * WARP_SIZE_ + thread_matrix_a_idx] = *reinterpret_cast<unsigned*>(&c_data_2);
        regfile_[(Rd + 1) * WARP_SIZE_ + thread_matrix_a_idx] = *reinterpret_cast<unsigned*>(&c_data_3);
      }
    }
  }
}

/**
 * @brief S2R instruction
 * @param Rd: the destination register
 * @param SR: store data type
 */
void GPU::SIM_S2R_INSTR(unsigned Rd, s_reg_t SR) {
  // S2R implementation
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; ++threadIdx) {
    unsigned data = 0;
    switch (SR) {
    case SR_LAINID:
      // SR_LANEID: thread index of the warp
      data = threadIdx;
      break;
    // The four following cases are not used in this project, so the implementation is fake
    case SR_TID_X:
      // SR_TID.X: threadIdx.x
      data = SR_TID_X;
      break;
    case SR_TID_Y:
      // SR_TID.Y: threadIdx.y
      data = SR_TID_Y;
      break;
    case SR_CTAID_X:
      // SR_CTAID.X: blockIdx.x
      data = SR_CTAID_X;
      break;
    case SR_CTAID_Y:
      // SR_CTAID.Y: blockIdx.y
      data = SR_CTAID_Y;
      break;
    default:
      // Invalid SR value
      break;
    }
    regfile_[Rd * WARP_SIZE_ + threadIdx] = data;
  }
}

/**
 * @brief IMAD instruction
 * @param Rd: the first destination register
 * @param Ra: the source register for multiplication
 * @param Sb: the source register for multiplication
 * @param Sc: the source register for addition
 * @param wide: 1 if the data is 64-bit, 0 if the data is 32-bit
 * @param fmt: 1 if the data is signed, 0 if the data is unsigned
 */
void GPU::SIM_IMAD_INSTR(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc, bool wide, unsigned fmt) {
  // IMAD implementation
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
    unsigned & ra_data = regfile_[Ra * WARP_SIZE_ + threadIdx];
    unsigned & sb_data = regfile_[Sb * WARP_SIZE_ + threadIdx];
    if (wide) {
      if (fmt) {
        int64_t data;
        data = ((int64_t)ra_data * (int64_t)sb_data) & 0xffffffffffffffff;
        unsigned &sc_data_0 = regfile_[Sc * WARP_SIZE_ + threadIdx];
        unsigned &sc_data_1 = regfile_[(Sc + 1) * WARP_SIZE_ + threadIdx];
        int64_t sc_data = concat(sc_data_1, sc_data_0);
        int64_t rd_data = data + sc_data;

        unsigned &rd_data_0 = regfile_[Rd * WARP_SIZE_ + threadIdx];
        unsigned &rd_data_1 = regfile_[(Rd + 1) * WARP_SIZE_ + threadIdx];
        split((uint64_t)rd_data, rd_data_1, rd_data_0);
      } else {
        uint64_t data;
        data = ((uint64_t)ra_data * (uint64_t)sb_data) & 0xffffffffffffffff;
        unsigned &sc_data_0 = regfile_[Sc * WARP_SIZE_ + threadIdx];
        unsigned &sc_data_1 = regfile_[(Sc + 1) * WARP_SIZE_ + threadIdx];
        uint64_t sc_data = concat(sc_data_1, sc_data_0);
        uint64_t rd_data = data + sc_data;

        unsigned &rd_data_0 = regfile_[Rd * WARP_SIZE_ + threadIdx];
        unsigned &rd_data_1 = regfile_[(Rd + 1) * WARP_SIZE_ + threadIdx];
        split(rd_data, rd_data_1, rd_data_0);
      }
    } else {
      if (fmt) {
        int data;
        data = ((int)ra_data * (int)sb_data) & 0xffffffff;
        int sc_data = (int)regfile_[Sc * WARP_SIZE_ + threadIdx];
        regfile_[Rd * WARP_SIZE_ + threadIdx] = data + sc_data;
      } else {
        unsigned data;
        data = ((unsigned)ra_data * (unsigned)sb_data) & 0xffffffff;

        unsigned &sc_data = regfile_[Sc * WARP_SIZE_ + threadIdx];
        regfile_[Rd * WARP_SIZE_ + threadIdx] = data + sc_data;
      }
    }
  }
}

void GPU::SIM_LOP3_INSTR(
  unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc,
  unsigned imm
) {
  // for: warp execuation
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
    // LOP3 implementation
    unsigned &ra_data = regfile_[Ra * WARP_SIZE_ + threadIdx];
    unsigned &sb_data = regfile_[Sb * WARP_SIZE_ + threadIdx];
    unsigned &sc_data = regfile_[Sc * WARP_SIZE_ + threadIdx];
    unsigned data = 0;
    if (imm & 0x01) data |= (~ra_data) & (~sb_data) & (~sc_data);
    if (imm & 0x02) data |= (~ra_data) & (~sb_data) & (sc_data);
    if (imm & 0x04) data |= (~ra_data) & (sb_data) & (~sc_data);
    if (imm & 0x08) data |= (~ra_data) & (sb_data) & (sc_data);
    if (imm & 0x10) data |= (ra_data) & (~sb_data) & (~sc_data);
    if (imm & 0x20) data |= (ra_data) & (~sb_data) & (sc_data);
    if (imm & 0x40) data |= (ra_data) & (sb_data) & (~sc_data);
    if (imm & 0x80) data |= (ra_data) & (sb_data) & (sc_data);
    regfile_[Rd * WARP_SIZE_ + threadIdx] = data;
  }
}

/**
 * @brief SHF instruction
 * @param Rd: the destination register
 * @param Ra: the register containing the low 32 bits of the value to be shifed
 * @param Sb: the register containing the shift bits
 * @param Sc: the register containing the high 32 bits of the value to be shifed
 * @param dir: the direction of the shift (0: left, 1: right)
 * @param maxshift: logical shift (1) or arithmetic shift (0)
 * @param HI: shift bits plus 32 (1) or not (0)
 */
void GPU::SIM_SHF_INSTR(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc, bool dir, bool maxshift, bool HI) {
  // SHF implementation
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
    unsigned & ra_data = regfile_[Ra * WARP_SIZE_ + threadIdx];
    unsigned & sc_data = regfile_[Sc * WARP_SIZE_ + threadIdx];
    unsigned & sb_data = regfile_[Sb * WARP_SIZE_ + threadIdx];
    uint64_t val = concat(sc_data, ra_data);
    unsigned shift = HI ? (sb_data + 32) : sb_data;
    uint64_t data;
    if (maxshift) {
      // logical shift
      if (dir) {
        // right shift
        data = val >> shift;
      }
      else {
        data = val << shift;
      }
    } else {
      // arithmetic shift
      if (dir) {
        data = (int64_t)val >> shift;
      }
      else {
        data = (int64_t)val << shift;
      }
    }
    regfile_[Rd * WARP_SIZE_ + threadIdx] = (data & 0xffffffff);
  }
}

/**
 * @brief CS2R instruction
 * @param Rd: the destination register
 * @param SR: store data type (SRZ)
 */
void GPU::SIM_CS2R_INSTR(unsigned Rd, s_reg_t SR) {
  // S2R implementation
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; ++threadIdx) {
    if (SR != SRZ)
      printf("CS2R error: SR != SRZ\n");
    // 64-bit 0
    regfile_[Rd * WARP_SIZE_ + threadIdx] = 0;
    regfile_[(Rd + 1) * WARP_SIZE_ + threadIdx] = 0;
  }
}

void GPU::SIM_MOV_INSTR(unsigned Rd, unsigned Sc) {
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; ++threadIdx) {
    regfile_[Rd * WARP_SIZE_ + threadIdx] = Sc;
  }
}

void GPU::SIM_LEA_INSTR(bool HI, bool X, unsigned Rd, unsigned Ra, unsigned Sb, 
                        unsigned imm, unsigned Sc, unsigned Pd0, unsigned Ps0) {
  // for: warp execuation
  for (int threadIdx = 0; threadIdx < WARP_SIZE_; threadIdx++) {
    // LEA implementation
    unsigned ra_data = regfile_[Ra * WARP_SIZE_ + threadIdx];
    unsigned sb_data = regfile_[Sb * WARP_SIZE_ + threadIdx];
    uint64_t sc_data = regfile_[Sc * WARP_SIZE_ + threadIdx];
    uint64_t data = ra_data;
    if (Sc != 255) {
      uint64_t data = ((uint64_t)sc_data << 32) | ra_data;
    }
    if (HI)
      data = data >> (32 - imm);
    else
      data = data << imm;
    data += sb_data;
    if (X) data += pregfile_[Ps0 * WARP_SIZE_ + threadIdx];
    if (Pd0 != 7)
      pregfile_[Pd0 * WARP_SIZE_ + threadIdx] = ((data >> 32) & 0x1);
    data &= 0xffffffff;
    regfile_[Rd * WARP_SIZE_ + threadIdx] = data;
  }
}

void GPU::SIM_EXIT_INSTR() {
  // EXIT implementation

  // Free any dynamically allocated memory
  delete[] memory_;
  delete[] regfile_;
  delete[] pregfile_;
  allocated_size_ = 0;
  memory_size_ = 0;
  // Close any open files or connections
  // (if applicable)

  // Perform any other necessary shutdown tasks
  // (if applicable)
}

void simMalloc(void **ptr, size_t size, GPU &volta) {
  // sim cudaMalloc
  // Request GPU memory
  if (volta.allocated_size_ + size > volta.memory_size_) {
    // Not enough memory available, throw an exception or return an error code
    throw std::bad_alloc();
  }

  // Allocate memory on the GPU
  *ptr = (void*)volta.memory_ + volta.allocated_size_;
  volta.allocated_size_ += size;
}

inline int check_in_GPU(void *ptr, size_t count, GPU &volta) {
  // Check that the whole memory block is within the bounds of the GPU memory
  if (
    (unsigned char*)ptr >= (unsigned char*)(volta.memory_) &&
    (unsigned char*)ptr + count < (unsigned char*)(volta.memory_) + volta.allocated_size_
  )
    return 1;
  return 0;
}

void simMemcpy(void *dst, void *src, size_t count, enum cudaMemcpyKind kind,
               GPU &volta) {
  // sim cudaMemcpy
  // memcpy host memory to class GPU memory or
  // memcpy class GPU memory to host memory
  // Check that the destination and source pointers are within the bounds of the GPU memory
  if (
    (kind == MemcpyHostToDevice && check_in_GPU(dst, count, volta)) ||
    (kind == MemcpyDeviceToHost && check_in_GPU(src, count, volta))
  ) {
    // Copy memory between the host and the GPU
    memcpy(dst, src, count);
  } else {
    // Invalid destination or source pointer, throw an exception or return an error code
    throw std::invalid_argument("Invalid memory address");
  }
}

void wmma_kernel(__half *a, __half *b, float *c, float *d, dim3 &gridDim,
                 dim3 &blockDim, GPU &volta) {
  // device kernel function
  // gridDim & blockDim
  const uint64_t c_0_160 = (uint64_t) a;
  const uint64_t c_0_168 = (uint64_t) b;
  const uint64_t c_0_170 = (uint64_t) c;
  const uint64_t c_0_178 = (uint64_t) d;
  // // volta.SIM_IMAD_INSTR(1, 255, 255, c_0_28, 0, 0);
  volta.SIM_S2R_INSTR(24, SR_LAINID);
  // print_reg(volta, 24);
  volta.SIM_MOV_INSTR(1, 0x10);
  // volta.SIM_IMAD_INSTR(1, 255, 255, )
  volta.SIM_IMAD_INSTR(13, 255, 255, 1, 0, 0);

  volta.SIM_MOV_INSTR(1, 0x2);

  volta.SIM_SHF_INSTR(23, 255, 1, 24, 1, 1, 1);
  // print_reg(volta, 23);
  volta.SIM_MOV_INSTR(1, 0x4);
  volta.SIM_SHF_INSTR(22, 0, 1, 24, 1, 1, 1);
  // print_reg(volta, 22);
  // volta.SIM_MOV_INSTR(2, 0xc0);
  volta.SIM_MOV_INSTR(1, 0x3);
  volta.SIM_LOP3_INSTR(23, 23, 1, 255, 0xc0);
  // print_reg(volta, 23);
  volta.SIM_LOP3_INSTR(24, 24, 1, 255, 0xc0);
  // print_reg(volta, 24);
  volta.SIM_MOV_INSTR(1, 0x1);
  volta.SIM_LOP3_INSTR(3, 22, 1, 255, 0xc0);
  // print_reg(volta, 3);
  volta.SIM_MOV_INSTR(1, 0x8);
  // print_reg(volta, 23);
  volta.SIM_IMAD_INSTR(5, 23, 1, 255, 0, 0);
  volta.SIM_MOV_INSTR(1, 0x1);
  volta.SIM_SHF_INSTR(0, 255, 1, 23, 1, 1, 1);
  // print_reg(volta, 0);
  volta.SIM_MOV_INSTR(1, 0x8);
  volta.SIM_LOP3_INSTR(2, 5, 1, 24, 0xe2);
  // print_reg(volta, 2);
  volta.SIM_MOV_INSTR(1, 0x2);
  volta.SIM_IMAD_INSTR(4, 0, 1, 3, 0);
  // print_reg(volta, 4);
  volta.SIM_MOV_INSTR(1, 0x4);
  volta.SIM_IMAD_INSTR(3, 3, 1, 2, 0);
  // print_reg(volta, 3);
  volta.SIM_IMAD_INSTR(2, 4, 1, 255, 0, 0);
  // print_reg(volta, 2);
  volta.SIM_MOV_INSTR(1, 0x2);
  volta.SIM_IMAD_INSTR(12, 3, 1, 255, 0, 0);
  // print_reg(volta, 12);
  volta.SIM_IMAD_INSTR(3, 255, 255, 255, 0, 0);
  // print_reg(volta, 3);
  // store the address of a to some register, assume to be 100-101
  volta.SIM_MOV_INSTR(100, (unsigned)(c_0_160 & 0xffffffff));
  volta.SIM_MOV_INSTR(101, (unsigned)(c_0_160 >> 32));
  volta.SIM_IMAD_INSTR(12, 12, 13, 100, 1, 0);
  // print_reg(volta, 12);
  // print_reg_val(volta, 12);
  volta.SIM_MOV_INSTR(1, 0x10);
  volta.SIM_IMAD_INSTR(2, 24, 1, 2, 1, 0);
  // print_reg(volta, 2);
  volta.SIM_LDG_INSTR(1, 128, 16, 12, 0);
  // print_reg(volta, 16);
  // print_load(volta, 16);
  volta.SIM_MOV_INSTR(1, 0x1);
  volta.SIM_MOV_INSTR(100, (unsigned)(c_0_168 & 0xffffffff));
  volta.SIM_MOV_INSTR(101, (unsigned)(c_0_168 >> 32));
  // print_reg(volta, 2);
  // printf("0x%lx\n", (uint64_t)(c_0_168));
  volta.SIM_LEA_INSTR(false, false, 26, 2, 100, 1, 255, 0);
  // print_reg(volta, 3);
  volta.SIM_LEA_INSTR(true, true, 27, 2, 101, 1, 3, 7, 0);

  // print_reg_val(volta, 26);
  volta.SIM_CS2R_INSTR(4, SRZ);
  volta.SIM_LDG_INSTR(true, 128, 12, 12, 0x10);
  // print_load(volta, 12, 16, 128);
  volta.SIM_LDG_INSTR(true, 64, 2, 26, 0);
  volta.SIM_CS2R_INSTR(6, SRZ);
  volta.SIM_CS2R_INSTR(8, SRZ);
  volta.SIM_CS2R_INSTR(10, SRZ);
  volta.SIM_LDG_INSTR(true, 64, 20, 26, 0x80);
  // print_load(volta, 20, 16, 64);
  volta.SIM_MOV_INSTR(1, 0x4);
  volta.SIM_IMAD_INSTR(25, 22, 1, 255, 0, 0);
  // volta.SIM_MOV_INSTR(80, 0xe2);
  volta.SIM_LOP3_INSTR(24, 25, 1, 24, 0xe2);
  // print_load(volta, 16);
  // print_load(volta, 12);
  // print_load(volta, 2, 16, 64);
  // print_load(volta, 1, 16, 64);
  volta.SIM_HMMA_INSTR_STEP0(4, 16, 2, 4);
  // print_hmma_step0(volta, 4);
  
  // after step 0

  volta.SIM_HMMA_INSTR_STEP1(6, 16, 2, 6);
  // print_hmma_step0(volta, 6);
  // return ;
  
  volta.SIM_HMMA_INSTR_STEP2(8, 16, 2, 8);
  volta.SIM_HMMA_INSTR_STEP3(10, 16, 2, 10);

  volta.SIM_HMMA_INSTR_STEP0(4, 18, 20, 4);
  volta.SIM_LDG_INSTR(true, 64, 16, 26, 0x100);
  volta.SIM_LDG_INSTR(true, 64, 2, 26, 0x180);
  volta.SIM_MOV_INSTR(80, 0x2);
  volta.SIM_LOP3_INSTR(25, 24, 80, 255, 0xc0);
  volta.SIM_MOV_INSTR(80, 0x8);
  volta.SIM_IMAD_INSTR(23, 23, 80, 255, false, 0);
  volta.SIM_MOV_INSTR(81, 0x5);
  volta.SIM_LOP3_INSTR(24, 24, 81, 255, 0xc0);
  // volta.SIM_MOV_INSTR(80, 0x5);
  volta.SIM_IMAD_INSTR(22, 0, 80, 25, 0, 1);
  volta.SIM_LOP3_INSTR(25, 23, 80, 24, 0xe2);
  volta.SIM_IMAD_INSTR(23, 255, 255, 255, 0, 0);

  volta.SIM_HMMA_INSTR_STEP1(6, 18, 20, 6);
  volta.SIM_HMMA_INSTR_STEP2(8, 18, 20, 8);
  volta.SIM_MOV_INSTR(80, 0x10);
  volta.SIM_IMAD_INSTR(22, 25, 80, 22, true, 0);
  volta.SIM_HMMA_INSTR_STEP3(10, 18, 20, 10);

  volta.SIM_HMMA_INSTR_STEP0(4, 12, 16, 4);
  volta.SIM_HMMA_INSTR_STEP1(6, 12, 16, 6);
  volta.SIM_HMMA_INSTR_STEP2(8, 12, 16, 8);
  volta.SIM_HMMA_INSTR_STEP3(10, 12, 16, 10);

  volta.SIM_HMMA_INSTR_STEP0(4, 14, 2, 4);
  volta.SIM_MOV_INSTR(100, (unsigned)(c_0_170 & 0xffffffff));
  volta.SIM_MOV_INSTR(101, (unsigned)(c_0_170 >> 32));

  volta.SIM_LEA_INSTR(false, false, 12, 22, 100, 0x2, 255, 0);

  volta.SIM_HMMA_INSTR_STEP1(6, 14, 2, 6);
  volta.SIM_HMMA_INSTR_STEP2(8, 14, 2, 8);
  volta.SIM_HMMA_INSTR_STEP3(10, 14, 2, 10);
  volta.SIM_LEA_INSTR(true, true, 13, 22, 101, 0x2, 23, 7, 0);

  volta.SIM_STG_INSTR(true, 64, 4, 12, 0);
  volta.SIM_STG_INSTR(true, 64, 6, 12, 0x80);
  volta.SIM_STG_INSTR(true, 64, 8, 12, 0x10);
  volta.SIM_STG_INSTR(true, 64, 10, 12, 0x90);
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 16; ++j) {
      printf("%f ", c[i * 16 + j]);
    }
    printf("\n");
  } 
  // volta.SIM_EXIT_INSTR();

  // print_load(volta, 2, 16, 64);
  // print_reg_val(volta, 16);
  // 16 x 16 mma

  // load a, b, c to gpu memory
  // thread group 0, 2 load a[0:3], 
  // thread group 4, 6 load a[4:7]
  // thread group 1, 3 load a[8:11]
  // thread group 5, 7 load a[12:15]
  // volta.SIM_LDG_INSTR(1, 128, 2, ((unsigned int)a / 4), 0);
  // volta.SIM_LDG_INSTR(1, 128)
  // volta.SIM_IMAD_INSTR(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc, unsigned fmt, bool wide);  // SASS: IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
  // add instruction you need,sim_imad_instr() is just an example
}

void gemm(float *a, float *b, float *c, float *d) {
  // host function gemm
  // mxnxk=? According to the pytorch source code, find the matrix size of the
  // conv2d operator of Resnet18 when doing matrix multiplication after
  // im2col. Remember to turn off cudnn and use the cublas library We need to
  // considerthe settings of blockDim and gridDim, we limit only one warp, how
  // to slicethe matrix to a matrix of 16*16*16 size, what to do if it cannot
  // be divisible evenly
}

void im2col() {
  // ref caffe
}
void conv() {}

int main() {
//   float a_list[16] = {pow(2, -20), 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5};
//   float b_list[16] = {0, -0.1, 0.2, -0.3, 0.4, -0.5, 0.6, 0.7, 0.8, -0.9, 1, 1.1, 1.2, -1.3, 1.4, 1.5};
//   float c_list[16];

//   __half ha_list[16], hb_list[16];

//   for (int i = 0; i < 16; i++) {
//     ha_list[i] = __float2half(a_list[i]);
//     hb_list[i] = __float2half(b_list[i]);
//     std::cout << a_list[i] << " " << __half2float(ha_list[i]) << std::endl;
//   }

//   for (int i = 0; i < 16; i++) {
//     c_list[i] = a_list[i] * b_list[i];
//     float hc = ha_list[i] * hb_list[i];
//     printf("%f %f %f %f\n", c_list[i], hc, c_list[i] - hc, __half2float(__float2half(c_list[i])));
//   }
  vector<__half> a(16 * 16), b(16 * 16);
  vector<float> c(16 * 16);
  for (int i = 0; i < 16; i++) {
    for (int j = 0; j < 16; j++) {
      a[i * 16 + j] = __float2half(i * 16 + j);
      b[i * 16 + j] = __float2half(- i * 16 - j);
    }
  }

  int size = 16 * 16;
  __half *d_a, *d_b;
  float* d_c;
  float * d_d;
  GPU volta;
  simMalloc((void**)(&d_a), sizeof(__half) * size, volta);
  simMalloc((void**)(&d_b), sizeof(__half) * size, volta);
  simMalloc((void**)(&d_c), sizeof(float) * size, volta);
  simMalloc((void**)(&d_d), sizeof(float) * size, volta);
  simMemcpy(d_a, a.data(), sizeof(__half) * size, MemcpyHostToDevice, volta);
  simMemcpy(d_b, b.data(), sizeof(__half) * size, MemcpyHostToDevice, volta);
  dim3 grid = {1, 1, 1};
  dim3 block = {1, 1, 1};
  wmma_kernel(d_a, d_b, d_c, d_d, grid, block, volta);
  // for (int i = 0; i < 16; i++) {
  //   for (int j = 0; j < 16; j++) {
  //     printf("%f ", d_c[i * 16 + j]);
  //   }
  //   printf("\n");
  // }
  // unsigned a = 0x3f800000;
  // unsigned b = 0x10000001;
  // uint64_t c = concat(a, b);
  // printf("%x\n%x\n%lx\n", a, b, c);
  // printf("%x\n%x\n", (uint64_t)a, (uint64_t)b);

  // volta.SIM_LDG_INSTR(1, 64, 0, 0, 0);
  // volta.SIM_STG_INSTR(1, 64, 0, 0, 0);
}
