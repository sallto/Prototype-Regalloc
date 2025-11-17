# Basic IR
function condbr3

block b0:
  op defs=%v0
  op defs=%v1
  op defs=%v2
  op uses=%v0
  jmp b1,b2


block b1:
  op uses=%v1,%v2 defs=%v4
  jmp b2

block b2:
  phi %v6 [b0, %v1, b1, %v4]
  op uses=%v2,%v6 defs=%v7
  op uses=%v7