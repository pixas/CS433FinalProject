#ifndef _TENSORE_CORE_H_
#define _TENSORE_CORE_H_
#include <stdint.h>
#include <string.h>

#include <iostream>
#include <vector>
using std::vector;

enum cudaMemcpyKind {
  MemcpyHostToDevice = 0, /**< Host   -> Device */
  MemcpyDeviceToHost = 1  /**< Device -> Host */
};

enum s_reg_t { SRZ = 0, SR_LAINID, SR_TID_X, SR_TID_Y, SR_CTAID_X, SR_CTAID_Y };

struct dim3 {
  unsigned int x, y, z;
  constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
      : x(vx), y(vy), z(vz) {}
};

struct __half {
 protected:
  unsigned short __x;

 public:
  __half() = default;
  __half(unsigned short x) {__x = x;}
  unsigned short getX() const { return __x; }
  void setX(unsigned short x) { __x = x; }
};

extern __half __float2half(const float &a);

extern float __half2float(const __half &a);

extern float operator*(const __half &lh, const __half &rh);

class GPU {
 public:
  GPU();
  void SIM_LDG_INSTR(bool E, unsigned sz, unsigned Rd, unsigned Ra, unsigned imm);
  void SIM_STG_INSTR(bool E, unsigned sz, unsigned Rd, unsigned Ra, unsigned imm);
  void SIM_HMMA_INSTR_STEP0(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc, unsigned a_fmt=1, unsigned b_fmt=1);
  void SIM_HMMA_INSTR_STEP1(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc, unsigned a_fmt=1, unsigned b_fmt=1);
  void SIM_HMMA_INSTR_STEP2(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc, unsigned a_fmt=1, unsigned b_fmt=1);
  void SIM_HMMA_INSTR_STEP3(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc, unsigned a_fmt=1, unsigned b_fmt=1);
  void SIM_S2R_INSTR(unsigned Rd, s_reg_t SR);
  void SIM_IMAD_INSTR(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc, bool wide, unsigned fmt=1);
  void SIM_LOP3_INSTR(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc,
                      unsigned imm);
  void SIM_MOV_INSTR(unsigned Rd, unsigned Sc);
  void SIM_SHF_INSTR(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc, bool dir, bool maxshift, bool HI);
  void SIM_CS2R_INSTR(unsigned Rd, s_reg_t SR);
  void SIM_LEA_INSTR(bool HI, bool X, unsigned Rd, unsigned Ra, unsigned Sb, 
                     unsigned imm, unsigned Sc = 255,unsigned Pd0 = 7, unsigned Ps0 = 7);
  void SIM_EXIT_INSTR();
  unsigned *memory_;
  uint64_t allocated_size_;
  uint64_t memory_size_;
  unsigned total_thread() {return WARP_SIZE_;}
  unsigned reg_content(unsigned Rid, int t) {return regfile_[Rid * WARP_SIZE_ + t];}
 private:
  // unsigned warpNum_;
  unsigned *regfile_;
  bool *pregfile_;
  const unsigned WARP_SIZE_ = 32;
};
uint64_t concat(unsigned high, unsigned low);
extern void simMalloc(void **ptr, size_t size, GPU &volta);

extern void simMemcpy(void *dst, void *src, size_t count,
                      enum cudaMemcpyKind kind, GPU &volta);

extern void wmma_kernel(__half *a, __half *b, float *c, float *d, dim3 &gridDim,
                        dim3 &blockDim, GPU &volta);

extern void gemm(float *a, float *b, float *c, float *d);
#endif
