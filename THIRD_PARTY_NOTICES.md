# Third-party notices

This application contains the vendored AIGM Detector Python package at commit `83db9aca8215db53dc539e217414e18ffdb633ad`, and uses the `awsaf49/sonics-spectttra-alpha-120s` pretrained checkpoint, downloaded on first use.

## AIGM Detector source — MIT

Copyright (c) 2026 shoegazerstella

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## SONICS code and checkpoint — MIT

The model card lists the checkpoint as MIT. The official repository license lists code and models as MIT.

Copyright (c) 2023 awsaf49

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

The SONICS training dataset has a separate CC BY-NC 4.0 license. This notice records that provenance; it does not resolve whether commercial use of the trained checkpoint is permitted under all applicable terms.

## Runtime packages and FFmpeg

`setup_windows.bat` installs the listed Python dependencies into a virtual environment. Their license files are provided by their respective package distributions. The `imageio-ffmpeg` Python wrapper is BSD-2-Clause; its installed FFmpeg binary reports a GPLv3+ configuration (`--enable-gpl --enable-version3`). FFmpeg's own license and redistribution requirements apply to that binary. Before distributing a commercial installer, verify the exact FFmpeg build and provide its required source and notices, or replace it with a commercially suitable LGPL-only build.
