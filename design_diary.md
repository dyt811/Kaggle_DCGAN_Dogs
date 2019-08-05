2019-07-16T185716EST

Thinking of training a better discriminator, I think I will maybe also incorporate null pictures. I.E. help train the discrimnator to ensure that it also understand that SOMETHINGS are not dogs. 

Using old trust script, I can download some animals e.g. cats, office, random images to prevent discriminator from seeing the world full of dogs.

A more effective disciminiator will undoubtedly improve generator performance as they are trained in pair. So the more exotic shxt discrimnator see, the better generator will have to come up with to fool the discriminator ideally. 

We will see how easy that is to do in practice. 

2019-07-16T194216EST
Quite an elegantly and simply crafted DC gan that is barely a few hundred lines lone. I could add so much more to it: inception module for better scale invariant discrimination, more supplementary random data that will ensure higher specificity. 

2019-07-22T0925EST
So I have trained on this BigBeast of 2070x2 for about 48 hours and what I observe is that the discriminator has the job too good and it failed to guide the generator. 

As I was reading around yesterday, some tips were saying that it might be good to RANDOMLY force discriminators to make some mistakes. Some quick improvement: 1) adding TensorBoard integration. 2) force discriminator to be imperfect and almost semi-incompetent. Another important tips I did notice was that the discriminator loss was florring on the fake images suggesting a nash equilibrium was reached where regardless of what generator makes, discriminator nailed the classification and resulted in edge cases. 

2019-07-30T1413EST
Time is running short. I managed to inplement the SOFT label to GAN to ensure that it is not min-maxing in a local extreme and produce mode collapse. About 18h into the training,the mode collapse happened. Resulted in VERY clear temporal pattern from the generated images. ffmpeg -f image2 -pattern_type glob -i 'time-lapse-files/*.JPG'