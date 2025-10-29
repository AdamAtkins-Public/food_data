## Data processing

### Initial Condition

The data is initially on a collection of typical receipt paper.

We want a digital image of the data on which we will use CV to extract information.

I use my phone camera to take images of the receipts.

### Image preprocessing

The receipts contain text that we are not interested in, so we crop the image to the areas that contain our desired information.

I manually crop the images using MS Paint to limit noise.

### Image processing

The next step is to use CV to extract the information from the receipt images.

WIP:
I imagine that the process will require manual confirmation of information, but overall a reduction in time for data entry.

I plan to generate a collection of the original preprocessed image, the image with marked bounding boxes, and an textfile that contains the receipts' extracted text.