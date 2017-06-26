---
layout: default
title:  "Transferring Musical 'Style' Without Learning It"
date:   2017-05-20 01:22:52 -0400
categories: jekyll update
author:
- Shaun Barry and Youngmoo Kim
---
This blog post is a summary and demonstration of a paper submission to the 2017 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics ([WASPAA][waspaa]), ["Transferring Musical Style Without Learning It Using Random Convolutions"][paper-link]. We introduce improvements to the [audio style transfer algorithmm][original-post] proposed by Dmitry Ulyanov and Vadim Levedev.
All code used to generate the examples is written in TensorFlow and can be found [here][code-link].

The Improvements we offer are:
1. Better representations and transfer of elements of rhythmic style.
2. The ability to use different lengths of audio for source content and target style.
3. Ability to optimize audio directly to a time domain representation (as opposed to the magnitude-frequency domain), so that the Griffin-Lim algorithm [[1]][griffin-lim] is not needed for phase reconstruction.
4. A naive (no prior training for source audio) method for source separation using this framework.

# **Background**

The problem we are attempting to solve is the transfer of musical style from one single example onto another using their raw audio representations, rather than symbolic representations like MIDI. To do so, we take the framework provided by Ulyanov and Lebedev and offer improvements for representing the long-term rhythmic structure of the audio.

The success of Ulyanov and Lebedev’s implementation over previous attempts is partly attributed to their use of the Short-Time-Fourier-Transform (STFT) which is convolved in time, but they treat each frequency bin as a color channel. Although we can plot the log-magnitude STFT of an audio signal as an image, there a lot of differences between how our eyes process images to how our ears process audio. For example, an object which is rotated in an image is still the same object, while rotating an audio signal along the time-frequency axes would drastically change how the signal is processed to your ear. Since with images we convolve kernels along the 2 spatial axes and not 1 color axis, it makes sense that we’d only want to convolve in time and not frequency for the STFT. As a result, Ulyanov and Lebedev’s method will find patterns in time corresponding to certain frequency bins.  But this method will not identify patterns in frequency, meaning that the representation of audio is not pitch invariant. This can easily be addressed by simply pitch-shifting the songs to the same (or similar sounding) keys and shifting tempos.

Since their implementation uses randomly initialized weights, it makes sense to use the STFT representation over the time-domain representation of an audio signal, since the patterns in frequency are in some ways easier to discern and perceive  than variations in a time-domain audio model such as Deepmind’s Wavenet [[2]][wavenet-link].

**So you're telling me we don't even have to train neural networks at all to do musical "style" transfer??? What's the catch?**

Well, the catch is that we need to use many more filters than is practical to train in typical neural networks. Ulyanov and Lebedev used 4096 filters with their 1x11 convolutional kernel. Here we use the same for the harmonic network structure and either 4096 filters for 1 rhythmic layer described in detail below, or 2048 filters for more than 1 in a dilated-residual structure. Even though the weights are not optimized, we can use many random filters to represent general patterns in time which are useful for representing musical style. 


# **Representing Rhythmic "Style"**
For our purposes, we can think of musical audio as being generally comprised of  3 separate components: harmony, rhythm, and timbre. The algorithm proposed by Ulyanov and Lebedev has already shown the ability to jointly represent and transfer elements of “harmonic” and “timbral” style, but fails to represent rhythmic components of the audio signal. It’s not difficult to see why this is the case: Given an STFT of an audio signal sampled at a rate of 22050 Hz with 1025 frequency bins and a hop length of 512, the 1x11 kernel will only span 1/4 of a second. A full measure measure of a song with a tempo of 120 bpm in 4/4 will last 2 seconds, which means a kernel better suited for capturing rhythmic structure should be at least 5-10x as long. 

<body>
	<center> 1x11 Convolutional layer on STFT
	<img src="/convolution_gif.gif">
</center>
</body>	

One way to correct this problem would be to simply increase the size of the convolutional kernel, but this is problematic since the size of the kernel starts to become very large. Also, if we use too long a time kernel (and high frequency resolution), we’ll actually end up with aliases of certain sections of the style audio instead of generalized stylistic information. Instead, we propose adding a second layer that is run in parallel and uses the mel-spectrum representation of the original signal instead of the entire STFT. The reason for this is it allows us separate loss terms for the transfer of harmonic and rhythmic style. The mel-spectrum itself can be modeled as a linear transformation that reduces the number of frequency bins. For the case where only 3 mel-frequency bins are used, we can think of each channel as the amount of energy in bass, mid, and treble frequency ranges. For the extreme case of only one mel-bin, the transformation is analogous to converting a color image to grayscale, since that transformation can also be modeled as a linear combination of the color channels.

Thus, our harmonic/timbral kernels are large along the frequency axis, but short in time, while our rhythmic kernels have reduced resolution in frequency and long-term structure in time. o represent very long term structure, we also show that we can use a standard 1D dilated, residual convolutional network to develop long-term statistics useful for generating textures.


<!--<body>
	<center> 1x50 Convolutional layer on Mel-spectrum with 8 bins
	<img src="/noise.gif">
</center>
</body>	-->

Earlier we mentioned that we could easily shift pitch and tempo to match for improved  quality of  style transfer. We also can use the mel-spectrum representation for the content as a way of making our style transfer algorithm invariant to key/tuning. This is particularly helpful if songs are in the same key, but slightly different tunings. If we reduce our 1025 frequency bins to 8 mel-bins, you can still obtain a clear enough representation of the signal to show what words are being sung or when rhythmic hits are happening in different frequency bands, but not enough information for the algorithm to care about which key the content audio is in.

To see an example of using the rhythmic content instead of the harmonic content, jump to example #1 below in the Style Transfer Experiments Section.


# **Using Different Lengths of Audio**
Since the loss function for style only depends on the number of filters used, and not the number of time bins, we can think of the L2 style loss as measuring the distance between the statistics of the target audio and the style audio. These statistics are represented by what is essentially the covariance matrix of the observations in the convolutional feature maps.

# **Using L-BFGS-B In Place of Griffin-Lim Algorithm**
Since we're already using the L-BFGS-B optimization algorithm, it seems redundant to use another iterative algorithm to restore the phase after our optimization is finished. We can skip the Griffin-lim algorithm altogether by modeling the log-magnitude STFT transformation as a separate "layer" in our neural network. We can model the real and imaginary parts of the STFT using separate convolutional layers. Below is the code used to achieve this:

{% highlight python %}
def get_logmagnitude_STFT(x_time, fs, n_dft, hop_length):

    # get DFT kernels
    w_ks = 2.0*np.pi * np.linspace(0, fs/2.0, n_dft//2+1)
    dft_real_kernels = np.array([[np.cos(w_k * n) for n in timesteps]
                                 for w_k in w_ks])
    dft_imag_kernels = np.array([[np.sin(w_k * n) for n in timesteps]
                                 for w_k in w_ks])

    # windowing DFT filters
    dft_window = _hann(n_dft, sym=False)
    dft_window = dft_window.reshape((1, -1))
    dft_real_kernels = np.multiply(dft_real_kernels, dft_window)
    dft_imag_kernels = np.multiply(dft_imag_kernels, dft_window)

    # convolve to get log-magnitude STFT
    X_real = tf.nn.conv2d(x_time,
	    filter=dft_real_kernels,
	    strides=[1,hop_length,1,1],
	    padding="VALID")

    X_imag = tf.nn.conv2d(x_time,
    	filter=dft_imag_kernels,
    	strides=[1,hop_length,1,1],
    	padding="VALID")

    # get log-magnitude STFT
    X_mag = tf.sqrt(X_Real**2.0 + X_imag**2.0)
    X_logmag = tf.log1p(X_mag)
    return X_logmag
{% endhighlight %}

Since we use overlapping time windows (hop_length < n_dft), the gradient can be appropriately backpropogated to the correct time samples. 

This lets us optimize our target variable as the raw audio signal, rather than the log-magnitude STFT. Tests with reconstructing the audio based on the content loss alone show that this implementation not only gets better results in terms of phase error, but it also converges in less iterations and is faster in time, especially if a GPU is being used.

# **Listening to the Mel-reduced Content**

Before understanding how the mel-based representation helps us represent rhythmic style, let's listen to what the mel-reduced representations sound like at various stages of frequency bin reduction. Here we are only optimizing the rhythmic content loss with a sampling rate of 22.05 kHz and N_FFT = 2048.

<center>
	
<table>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Source</th>
		<td>
			<iframe width="560" height="315" src="https://www.youtube.com/embed/pIgZ7gMze7A?start=114&&end=134" frameborder="0" allowfullscreen></iframe>
		</td>
	</tr>
<tr>
	<th style="text-align: center; vertical-align: middle;">N_mels</th>
	<th style="text-align: center; vertical-align: middle;">Audio</th>
</tr>
<tr>
		<th style="text-align: center; vertical-align: middle;">512</th>
		<td>
		<audio controls>
      <source src="/contents/1_wakemeup-10-70_contentenergy-mel-512_kr-50_nresid-0_10000iters.wav">
      </audio>
		</td>
</tr>
<tr>
		<th style="text-align: center; vertical-align: middle;">256</th>
		<td>
		<audio controls>
      <source src="/contents/1_wakemeup-10-70_contentenergy-mel-256_kr-50_nresid-0_10000iters.wav">
      </audio>
		</td>
</tr>
<tr>
		<th style="text-align: center; vertical-align: middle;">128</th>
		<td>
		<audio controls>
      <source src="/contents/1_wakemeup-10-70_contentenergy-mel-128_kr-50_nresid-0_10000iters.wav">
      </audio>
		</td>
</tr>
<tr>
		<th style="text-align: center; vertical-align: middle;">64</th>
		<td>
		<audio controls>
      <source src="/contents/1_wakemeup-10-70_contentenergy-mel-64_kr-50_nresid-0_10000iters.wav">
      </audio>
		</td>
</tr>
<tr>
		<th style="text-align: center; vertical-align: middle;">32</th>
		<td>
		<audio controls>
      <source src="/contents/1_wakemeup-10-70_contentenergy-mel-32_kr-50_nresid-0_10000iters.wav">
      </audio>
		</td>
</tr>
<tr>
		<th style="text-align: center; vertical-align: middle;">16</th>
		<td>
		<audio controls>
      <source src="/contents/1_wakemeup-10-70_contentenergy-mel-16_kr-50_nresid-0_10000iters.wav">
      </audio>
		</td>
</tr>
<tr>
		<th style="text-align: center; vertical-align: middle;">8</th>
		<td>
		<audio controls>
      <source src="/contents/1_wakemeup-10-70_contentenergy-mel-8_kr-50_nresid-0_10000iters.wav">
      </audio>
		</td>
</tr>
<tr>
		<th style="text-align: center; vertical-align: middle;">4</th>
		<td>
		<audio controls>
      <source src="/contents/1_wakemeup-10-70_contentenergy-mel-4_kr-50_nresid-0_10000iters.wav">
      </audio>
		</td>
</tr>
<tr>
		<th style="text-align: center; vertical-align: middle;">2</th>
		<td>
		<audio controls>
      <source src="/contents/1_wakemeup-10-70_contentenergy-mel-2_kr-50_nresid-0_10000iters.wav">
      </audio>
		</td>
</tr>
	

</table>
	
</center>
# **Isolated Style Textures**

By only optimizing the style loss with no content loss, we can create audio textures like the ones shown in Ulyanov and Lebedev's original blog post. However, since we can represent longer-term structure in the statistics well, the textures have rhythmic structure similar to the original audio. Below is a detailed example comparing different lengths of kernels and types of layers. 

When we use more than one layer, we use residual and strided convolutions to increase our representational span even more. This architecture is actually very similar to the structure of the encoding stage in Wavenet auto-encoders. We use 2048 filters per convolutional layer, and we use a larger initial filter length of 50, while the additional residual blocks use a length of 25. When we use a sampling rate of 44.1 kHz (as most examples do), we only represent the reduced mel-binned STFT for frequencies below 11.025 kHz, since most frequencies above this will be correlated with the other frequencies below this threshold, and the harmonic style loss will be able to pick up this detail. Another way to think of the mel-spectrum representation is as a form of "rhythmic attention". 
<center>

<table>

	<!-- Crazy -->
	<tr>
		<th style="text-align: center; vertical-align: middle;"></th>
		<th style="text-align: center; vertical-align: middle;">Example 1</th>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Source</th>
		<td>
			<iframe width="560" height="315" src="https://www.youtube.com/embed/Qe500eIK1oA" frameborder="0" allowfullscreen></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Texture (Harmonic Loss Only, K_h = 5)</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="100%" height="166" scrolling="no" frameborder="no" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/327530413&amp;color=ff5500&amp;auto_play=false&amp;hide_related=false&amp;show_comments=true&amp;show_user=true&amp;show_reposts=false"></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Texture (Harmonic Loss Only, K_h = 10)</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="100%" height="166" scrolling="no" frameborder="no" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/327530560&amp;color=ff5500&amp;auto_play=false&amp;hide_related=false&amp;show_comments=true&amp;show_user=true&amp;show_reposts=false"></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Texture (Harmonic+Rhythmic, 512 mels, 1 Layer)</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="100%" height="166" scrolling="no" frameborder="no" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/327627912&amp;color=ff5500&amp;auto_play=false&amp;hide_related=false&amp;show_comments=true&amp;show_user=true&amp;show_reposts=false"></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Texture (Harmonic+Rhythmic, 512 mels, 4 Residual Blocks)</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="100%" height="166" scrolling="no" frameborder="no" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/327627948&amp;color=ff5500&amp;auto_play=false&amp;hide_related=false&amp;show_comments=true&amp;show_user=true&amp;show_reposts=false"></iframe>
		</td>
	</tr>

	<!-- Free Bird -->
	<tr>
		<th style="text-align: center; vertical-align: middle;"></th>
		<th style="text-align: center; vertical-align: middle;">Example 2</th>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Source</th>
		<td>
			<iframe width="560" height="315" src="https://www.youtube.com/embed/3GSbJkU9Lmw?start=0&&end=240" frameborder="0" allowfullscreen></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Texture</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="100%" height="166" scrolling="no" frameborder="no" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/326087328&amp;color=ff5500&amp;auto_play=false&amp;hide_related=false&amp;show_comments=true&amp;show_user=true&amp;show_reposts=false"></iframe>
		</td>
	</tr>
</table>



</center>

# **Style Transfer Experiments**
## **Examples Where Rhythmic Content Is Best**

Here are 2 examples of where using the content representation from the rhythm network works much better than the representation from the harmonic network since the two songs are in different keys/tunings. 
<center>

<table>
	<!--     Jacob Collier -->
	<tr>
		<th style="text-align: center; vertical-align: middle;"></th>
		<th style="text-align: center; vertical-align: middle;">Example 1</th>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Content</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="560" height="315" src="https://www.youtube.com/embed/EkPy18xW1j8?start=15&&end=45" frameborder="0" allowfullscreen></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Style</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="560" height="315" src="https://www.youtube.com/embed/cttFanV0o7c?start=5&&end=74" frameborder="0" allowfullscreen></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Result</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="100%" height="166" scrolling="no" frameborder="no" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/327630860&amp;color=ff5500&amp;auto_play=false&amp;hide_related=false&amp;show_comments=true&amp;show_user=true&amp;show_reposts=false"></iframe>
		</td>
	</tr>	
	<!--     IconABBA -->
	<tr>
		<th style="text-align: center; vertical-align: middle;"></th>
		<th style="text-align: center; vertical-align: middle;">Example 2</th>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Content</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="560" height="315" src="https://www.youtube.com/embed/cvChjHcABPA?start=38&&end=68" frameborder="0" allowfullscreen></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Style</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="560" height="315" src="https://www.youtube.com/embed/UxxajLWwzqY?start=35&&end=95" frameborder="0" allowfullscreen></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Result</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="100%" height="166" scrolling="no" frameborder="no" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/327628339&amp;color=ff5500&amp;auto_play=false&amp;hide_related=false&amp;show_comments=true&amp;show_user=true&amp;show_reposts=false"></iframe>
		</td>
	</tr>	
	<!--
	<tr>
		<th style="text-align: center; vertical-align: middle;">Result</th>
		<td style="text-align: center; vertical-align: middle;">
			<audio controls id="" preload="auto" >
			<source src="https://www.dropbox.com/s/b0o1l018evjxil9/1_content.wav?dl=1">
			</audio>
		</td>
	</tr>	
-->
</table>
	
</center>

## **Examples Where Harmonic Content Is Best**

When harmonic content is used, and there are a large amount of mel bins used (in this case 512), the result seems to resemble a mash-up of the two songs. 
<center>

<table>
	<!--     Won't get fooled again -->
	<tr>
		<th style="text-align: center; vertical-align: middle;"></th>
		<th style="text-align: center; vertical-align: middle;">Example 1</th>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Content</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="560" height="315" src="https://www.youtube.com/embed/SHhrZgojY1Q?start=440&&end=482" frameborder="0" allowfullscreen></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Style</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="560" height="315" src="https://www.youtube.com/embed/hLQl3WQQoQ0" frameborder="0" allowfullscreen></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Result</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="100%" height="166" scrolling="no" frameborder="no" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/326059376&amp;color=ff5500&amp;auto_play=false&amp;hide_related=false&amp;show_comments=true&amp;show_user=true&amp;show_reposts=false"></iframe>
		</td>
	</tr>	

	<!--     Africa -->
	<tr>
		<th style="text-align: center; vertical-align: middle;"></th>
		<th style="text-align: center; vertical-align: middle;">Example 2</th>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Content</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="560" height="315" src="https://www.youtube.com/embed/_MBRFbHHNzg?start=0&&end=30" frameborder="0" allowfullscreen></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Style</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="560" height="315" src="https://www.youtube.com/embed/FTQbiNvZqaY?start=4" frameborder="0" allowfullscreen></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Result</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="100%" height="166" scrolling="no" frameborder="no" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/326104554&amp;color=ff5500&amp;auto_play=false&amp;hide_related=false&amp;show_comments=true&amp;show_user=true&amp;show_reposts=false"></iframe>
		</td>
	</tr>	

	<!--     PsychoSocial -->
	<tr>
		<th style="text-align: center; vertical-align: middle;"></th>
		<th style="text-align: center; vertical-align: middle;">Example 3</th>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Content</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="560" height="315" src="https://www.youtube.com/embed/LB5YkmjalDg?start=28&&end=48" frameborder="0" allowfullscreen></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Style</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="560" height="315" src="https://www.youtube.com/embed/5abamRO41fE?start=60&&end=90" frameborder="0" allowfullscreen></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Result</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="100%" height="166" scrolling="no" frameborder="no" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/326147183&amp;color=ff5500&amp;auto_play=false&amp;hide_related=false&amp;show_comments=true&amp;show_user=true&amp;show_reposts=false"></iframe>
		</td>
	</tr>	
	<!--     Down Rodeo -->
	<tr>
		<th style="text-align: center; vertical-align: middle;"></th>
		<th style="text-align: center; vertical-align: middle;">Example 4</th>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Content</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="560" height="315" src="https://www.youtube.com/embed/IKyVYdIkwOQ?start=25&&end=55" frameborder="0" allowfullscreen></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Style</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="560" height="315" src="https://www.youtube.com/embed/do6Ki6kMq_o" frameborder="0" allowfullscreen></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Result</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="100%" height="166" scrolling="no" frameborder="no" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/327628752&amp;color=ff5500&amp;auto_play=false&amp;hide_related=false&amp;show_comments=true&amp;show_user=true&amp;show_reposts=false"></iframe>
		</td>
	</tr>	
	<!--     Beverly hills -->
	<tr>
		<th style="text-align: center; vertical-align: middle;"></th>
		<th style="text-align: center; vertical-align: middle;">Example 5</th>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Content</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="560" height="315" src="https://www.youtube.com/embed/HL_WvOly7mY?start=71&&end=91" frameborder="0" allowfullscreen></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Style</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="560" height="315" src="https://www.youtube.com/embed/aA5kIN2mr-o?start=113" frameborder="0" allowfullscreen></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Result</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="100%" height="166" scrolling="no" frameborder="no" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/327629912&amp;color=ff5500&amp;auto_play=false&amp;hide_related=false&amp;show_comments=true&amp;show_user=true&amp;show_reposts=false"></iframe>
		</td>
	</tr>	
</table>
</center>

## **Examples Where Either Content Representation Works Well**
For some examples, using either type of content works well. 
<center>

<table>

	<!--     Sweet Dreams -->
	<tr>
		<th style="text-align: center; vertical-align: middle;"></th>
		<th style="text-align: center; vertical-align: middle;">Example 1</th>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Content</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="560" height="315" src="https://www.youtube.com/embed/eRhg7qPLeN8?start=15&&end=45" frameborder="0" allowfullscreen></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Style</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="560" height="315" src="https://www.youtube.com/embed/SDTZ7iX4vTQ?start=4" frameborder="0" allowfullscreen></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Result</th>
		<td style="text-align: center; vertical-align: middle;">
		<iframe width="100%" height="166" scrolling="no" frameborder="no" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/326157068&amp;color=ff5500&amp;auto_play=false&amp;hide_related=false&amp;show_comments=true&amp;show_user=true&amp;show_reposts=false"></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Result (Rhythm Content)</th>
		<td style="text-align: center; vertical-align: middle;">
		<iframe width="100%" height="166" scrolling="no" frameborder="no" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/327650762&amp;color=ff5500&amp;auto_play=false&amp;hide_related=false&amp;show_comments=true&amp;show_user=true&amp;show_reposts=false"></iframe>
		</td>
	</tr>	
	<!--     Queen -->
	<tr>
		<th style="text-align: center; vertical-align: middle;"></th>
		<th style="text-align: center; vertical-align: middle;">Example 2</th>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Content</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="560" height="315" src="https://www.youtube.com/embed/fJ9rUzIMcZQ?start=220&&end=290" frameborder="0" allowfullscreen></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Style</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="560" height="315" src="https://www.youtube.com/embed/WSeNSzJ2-Jw?start=0&&end=116" frameborder="0" allowfullscreen></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Result</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="100%" height="166" scrolling="no" frameborder="no" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/326091390&amp;color=ff5500&amp;auto_play=false&amp;hide_related=false&amp;show_comments=true&amp;show_user=true&amp;show_reposts=false"></iframe>
		</td>
	</tr>	
	<tr>
		<th style="text-align: center; vertical-align: middle;">Result (Content Rhythm Version)</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="100%" height="166" scrolling="no" frameborder="no" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/327530219&amp;color=ff5500&amp;auto_play=false&amp;hide_related=false&amp;show_comments=true&amp;show_user=true&amp;show_reposts=false"></iframe>
		</td>
	</tr>
	<!--     Ghostbusters -->
	<tr>
		<th style="text-align: center; vertical-align: middle;"></th>
		<th style="text-align: center; vertical-align: middle;">Example 3</th>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Content</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="560" height="315" src="https://www.youtube.com/embed/Fe93CLbHjxQ?start=11&&end=62" frameborder="0" allowfullscreen></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Style</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="560" height="315" src="https://www.youtube.com/embed/4D7u5KF7SP8?start=0&&end=240" frameborder="0" allowfullscreen></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Result</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="100%" height="166" scrolling="no" frameborder="no" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/326090736&amp;color=ff5500&amp;auto_play=false&amp;hide_related=false&amp;show_comments=true&amp;show_user=true&amp;show_reposts=false"></iframe>
		</td>
	</tr>	
	<tr>
		<th style="text-align: center; vertical-align: middle;">Result (Content Rhythm Version)</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="100%" height="166" scrolling="no" frameborder="no" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/327628993&amp;color=ff5500&amp;auto_play=false&amp;hide_related=false&amp;show_comments=true&amp;show_user=true&amp;show_reposts=false"></iframe>
		</td>
	</tr>	
</table>
</center>
<!--
# **Naive Source Separation**

Given that we've shown the capability to represent musical statistics well, one could imagine a very simple algorithm for performing source separation as follows:
Given a signal, $$x$$, which can be decomposed into subsignals $$x_i, i \in [1,...,N]$$ and examples of each target decomposotion which have same musical statistical representation, $$G_i$$,
find all components $${x_i}$$ s.t.
1. $$ \sum_{i=1}^{N}{x_i} = x $$:
	
	$$ L_{Reconstruction} = L2(\sum_{i=1}^{N}{x_i}, x)$$
2. The statistics of each of the components are matched by the sound examples provided:
	
	$$L_{Style} = \sum_{i=1}^{N}{L2(S_i, G_i)}$$

<center>

<table>
	<tr>
		<th style="text-align: center; vertical-align: middle;"></th>
		<th style="text-align: center; vertical-align: middle;">Example 1</th>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Source</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="560" height="315" src="https://www.youtube.com/embed/hLQl3WQQoQ0?start=20&&end=30" frameborder="0" allowfullscreen></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Isolated Vocals (Target)</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="560" height="315" src="https://www.youtube.com/embed/xTchyZ6gsMI?start=20&&end=30" frameborder="0" allowfullscreen></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Isolated Vocals 2 (Target)</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="560" height="315" src="https://www.youtube.com/embed/byqTGIkmgAQ?start=0&&end=60" frameborder="0" allowfullscreen></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Isolated Piano (Target)</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="560" height="315" src="https://www.youtube.com/embed/0dGd6tPqv54?start=20&&end=30" frameborder="0" allowfullscreen></iframe>
		</td>
	</tr>
	<tr>
		<th style="text-align: center; vertical-align: middle;">Result: Isolated Vocals (Using Actual Isolation)</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="100%" height="166" scrolling="no" frameborder="no" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/326113798&amp;color=ff5500&amp;auto_play=false&amp;hide_related=true&amp;show_comments=true&amp;show_user=true&amp;show_reposts=false"></iframe>
		</td>
	</tr>	

	<tr>
	<th style="text-align: center; vertical-align: middle;">Result: Isolated Piano (Using Actual Isolation)</th>
	<td style="text-align: center; vertical-align: middle;">
		<iframe width="100%" height="166" scrolling="no" frameborder="no" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/326109894&amp;color=ff5500&amp;auto_play=false&amp;hide_related=true&amp;show_comments=true&amp;show_user=true&amp;show_reposts=false"></iframe>
	</td>
	</tr>	
	<tr>
		<th style="text-align: center; vertical-align: middle;">Result: Isolated Vocals (Using Set Fire)</th>
		<td style="text-align: center; vertical-align: middle;">
			<iframe width="100%" height="166" scrolling="no" frameborder="no" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/326130441&amp;color=ff5500&amp;auto_play=false&amp;hide_related=true&amp;show_comments=true&amp;show_user=true&amp;show_reposts=false"></iframe>
		</td>
	</tr>	

	<tr>
	<th style="text-align: center; vertical-align: middle;">Result: Isolated Piano (Using Set Fire)</th>
	<td style="text-align: center; vertical-align: middle;">
		<iframe width="100%" height="166" scrolling="no" frameborder="no" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/326139571&amp;color=ff5500&amp;auto_play=false&amp;hide_related=true&amp;show_comments=true&amp;show_user=true&amp;show_reposts=false"></iframe>
	</td>
	</tr>	
</table>
</center>-->

<div id="disqus_thread"></div>
<script>

/**
*  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
*  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables*/
/*
var disqus_config = function () {
this.page.url = PAGE_URL;  // Replace PAGE_URL with your page's canonical URL variable
this.page.identifier = PAGE_IDENTIFIER; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
};
*/
(function() { // DON'T EDIT BELOW THIS LINE
var d = document, s = d.createElement('script');
s.src = 'https://shaunbarry-github-io.disqus.com/embed.js';
s.setAttribute('data-timestamp', +new Date());
(d.head || d.body).appendChild(s);
})();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>

<script src="mathjax-config.js"></script>

<script type="text/javascript"
    src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>


[paper-link]: https://arXiv.org
[code-link]:   https://github.com/u/ShaunBarry/
[original-post]: https://dmitryulyanov.github.io/audio-texture-synthesis-and-style-transfer
[griffin-lim]: http://ieeexplore.ieee.org/document/1164317/
[waspaa]: http://www.waspaa.com/
[wavenet-link]: https://deepmind.com/blog/wavenet-generative-model-raw-audio/

