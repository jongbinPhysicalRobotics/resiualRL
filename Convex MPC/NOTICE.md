# 출처와 라이선스

이 폴더의 컨트롤러 구조 (보행 상태기계, 걸음 시간표, 접촉 관리, 착지점 계획, 스윙 · 디딤 다리 토크, condensed SRB MPC 조립) 는
아래 C++ 구현을 파이썬으로 옮긴 뒤 고쳐 쓴 것입니다.
그 위에 더한 것은 OA-MPC 선 접촉 제약, 해 자르기, 고정 반복 ADMM, 착지 이득 조정, 긴 걸음 주기 등 A30 설정입니다 (Q&A 9/26 ~ 10/1).

- **convex-mpc-biped** — https://github.com/ispaik06/convex-mpc-biped (commit 44e9618)

```
MIT License

Copyright (c) 2026 Inseong Paik

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

로봇 모델 (`models/mit_humanoid/`) 의 출처와 라이선스는 그 폴더의 `README.md` 와 `LICENSE_source` 에 있습니다.
