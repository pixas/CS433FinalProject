R24 = list(range(32))
R13 = [0x10 for _ in range(32)]
R23 = [R24_i >> 2 for R24_i in R24]
R22 = [R24_i >> 4 for R24_i in R24]

R23 = [R23_i & 0x3 for R23_i in R23]
R24 = [R24_i & 0x3 for R24_i in R24]
R3 = [R22_i & 0x1 for R22_i in R22]
R5 = [8 * R23_i for R23_i in R23]
R0 = [R23_i >> 1 for R23_i in R23]
R2 = []
for i in range(32):
    R2.append(
        (~R5[i] & ~0x8 & R24[i]) |
        (R5[i] & ~0x8 & R24[i]) |
        (R5[i] & 0x8 & ~R24[i]) |
        (R5[i] & 0x8 & R24[i])
    )
R4 = [R0[i] * 2 + R3[i] for i in range(32)]
R3 = [R3[i] * 4 + R2[i] for i in range(32)]
R2 = [R4_i * 4 for R4_i in R4]
R12 = [R3_i * 2 for R3_i in R3]
R3 = [0 for _ in range(32)]
R12 = [R12[i] * R13[i] for i in range(32)]
R2 = [R24[i] * 16 + R2[i] for i in range(32)]

print(R12)