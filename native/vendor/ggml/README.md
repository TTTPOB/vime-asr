# GGML CPU 编译头文件

这四个头文件与 GGML 两个共享库配对使用，原样来自 transcribe.cpp v0.4.0，commit `f1dcd15858c4bbba87ca97862b4e448b2cf15b8f` 的 `ggml/include/`。其 GGML 上游为 `d7cb574130e6f01ad25b3289685489200febcd74`，包含该项目的两个补丁。

`ggml_get_type_traits_cpu` 是内部 CPU 接口，不能单独更新头文件或二进制。重新选版本时验证量化 dot 的数据布局和实际转写。

源码许可见 [GGML MIT](LICENSE.ggml)，发布包许可见 [transcribe.cpp MIT](LICENSE.transcribe)。haswell CPU 库包含 Mozilla llamafile 的 SGEMM 实现，其 MIT 声明见 [LICENSE.llamafile](LICENSE.llamafile)。
