Clippie: A little inference-only implementation of CLIP
=======================================================

Clippie is a simple, CPU-based, pure [Numpy](https://numpy.org/) implementation
of (the forward pass of) OpenAI's [CLIP](https://openai.com/research/clip)
image and text encoding deep neural network.


Usage
-----

Once you've obtained a set of weights in Clippie's native format, you can use
it like so:

    >>> from clippie import load, encode_text, encode_image
    
    >>> # Load the model weights (actually memory maps them)
    >>> with open("path/to/ViT-B-32.weights", "rb") as f:
    ...     weights = load(f)
    
    >>> # Encode some text
    >>> text_vectors = encode_text([
    ...     "a toddler looking inside a computer",
    ...     "people walking along a mountain ridge",
    ...     "a beautiful lake",
    ... ], weights.text_encoder)
    >>> text_vectors.shape
    (3, 512)
    
    >>> # Encode some images
    >>> from PIL import Image
    >>> image_vectors = encode_image([
    ...     Image.open("toddler.jpg"),
    ...     Image.open("mountain.jpg"),
    ...     Image.open("lake.jpg"),
    ... ], weights.image_encoder)
    >>> image_vectors.shape
    (3, 512)
    
    >>> # Compute similarity scores
    >>> import numpy as np
    >>> text_vectors /= np.linalg.norm(image_vectors, axis=1, keepdims=True)
    >>> image_vectors /= np.linalg.norm(image_vectors, axis=1, keepdims=True)
    >>> similarity = text_vectors @ image_vectors.T
    
    >>> # Note that the matching text/image pairs (on the diagonal) have the
    >>> # highest values as you would hope.
    >>> similarity
    array([[0.29675007, 0.0999563 , 0.12603459],
           [0.09451606, 0.25567788, 0.18573087],
           [0.1604508 , 0.17910984, 0.2590417 ]], dtype=float32)


Generating a Weights File
-------------------------

The `clippie-convert-weights-file` script can be used to convert a [CLIP
weights file as used by the reference CLIP
implementation](https://github.com/openai/CLIP/blob/c5478aac7b9e007a2659d36b57ebe148849e542a/clip/clip.py#L36-L39)
into the [Clippie native format](./clippie/serialiser.py):

    $ pip install clippie[convert]  # Extra packages needed for weights file conversion
    $ clippie-convert-weights-file ~/.cache/clip/ViT-B-32.pt ViT-B-32.weights

The conversion script requires PyTorch to be installed. After conversion,
PyTorch is nolonger required.

The converted weights file will typically be larger than the source CLIP
weights file because it contains all values expanded into float32 which can be
directly memory mapped and used by Clippie.


Preemptive FAQ
--------------

**Why does Clippie exist?**

I wanted a decent search facility for my personal photo collection without
relying on a 3rd party serivce (e.g. Google Photos). Based on the impressive
results reported by
[various](https://mazzzystar.github.io/2022/12/29/Run-CLIP-on-iPhone-to-Search-Photos/)
other [projects](https://paulw.tokyo/post/real-time-semantic-search-demo/) it
became clear that OpenAI's CLIP model could work well, [in spite of this
usecase being explicitly
out-of-scope](https://github.com/openai/CLIP/blob/main/model-card.md#out-of-scope-use-cases).
In fact, in my experience so far, it works considerably better than Google
Photos' search function.

To ensure I could build my photo search system on something which would remain
stable for some years, I wanted to avoid using anything based on a cutting-edge
deep learning framework -- ruling out the reference implementation and other
open source options. I am not in this for the research: I just want the tasty,
tasty search results!

Finally, I've been looking for a reason to finally learn more about deep
learning and this was a good excuse. You'll hopefully find a little more detail
in the comments than you might otherwise expect as a result.


**Why not the reference implementation?**

By contrast with [CLIP's PyTorch-based reference
implementation](https://github.com/openai/CLIP), Clippie has only a few
comparatively light-weight and stable dependencies (chiefly
[Numpy](https://numpy.org/) and
[Pillow](https://pillow.readthedocs.io/en/stable/)). As such, the largest
download needed is a copy of the weights, not gigabytes of software.
Furthermore, unlike most deep learning libraries -- which cater to a fast
moving field -- all of the dependencies used have been stable and well
supported for many years and are likely to remain so indefinitely.

Separately, CLIP makes some slightly quirky choices in its implementation from
a software engineering point of view (e.g. its [odd vocabulary encoding
format](./clippie/scripts/convert_vocab_file.py). As such, Clippie does things
differently and hopefully more clearly.


**Why CPU only?**

The smallest ViT-B/32 model can process an image or text string on a modern CPU
in about a tenth of a second. This is plenty fast enough for (my) personal use
cases so no need to buy (or manage) a GPU!

NB: Clippie's Numpy based implementation runs approximately as fast as the
PyTorch-based reference implementation on a CPU. (The text encoder is
slightly faster, the image encoder is slightly slower. Its good enough for me
so I've not bothered to look into optimising it any further).


**Where can I download a Clippie-formatted weights file?**

Sorry, I don't have somewhere I can casually throw up multi-hundred megabyte
files so you'll have to convert the CLIP weights by hand (see notes above).


**Why another format to store model weights?**

Clippie uses its own custom on-disk format to store model weights. This format
supports using the weights directly memory mapped from disk. As a result,
whilst Clippie is idle, memory used by weights can be swapped out by the
operating system, preventing it from hogging memory when not needed.


**Why ViT-only?**

The CLIP authors reported that their [Vision
Transformer](https://arxiv.org/abs/2010.11929)-based image encoder approach
worked as well or better than other approaches tried. Since the text encoder
already uses a [Transformer](https://arxiv.org/abs/1706.03762), I'd already
done most of the work.


**Why all float32 only?**

Whilst some of CLIP's Vision Transformer weights are given as 16 bit values,
with the exception of some recent ARM systems, most CPUs only support efficient
float 32 arithmetic. This inflates memory usage somewhat but is faster on most
systems in practice.


**Why forward-pass only?**

I didn't need it. (Also, I have no interest in falling down the rabbit hole of
model training!)


**What about vector similarity search?**

Whilst various fancy [libraries](https://github.com/facebookresearch/faiss) and
[services](https://www.pinecone.io/) exist which implement fast approximate
nearest neighbour search for vast collections of vectors, they simply aren't
needed for my purposes. Numpy can brute-force search 100k vectors in about 50ms
on my laptop.


**Does this reuse any code from the CLIP reference implementation?**

This software is an independent reimplementation of CLIP and does not reuse, or
derive from its code. Where necessary it Clippie does deliberately makes the
same arbitrary design choices to ensure compatibility of weights.

Data files, including both model weights (not in this repository) and the
tokenizer vocabulary (included in this repository), are used directly from CLIP
by necessity. The vocabulary data is assumed to be [MIT licensed along with the
rest of the CLIP source
code](https://github.com/openai/CLIP/blob/main/LICENSE). The [license of the
model weights are unclear](https://github.com/openai/CLIP/issues/203) and they
are not redistributed here.
