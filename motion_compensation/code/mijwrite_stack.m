function mijwrite_stack(img,filename,color)

if nargin<3
    color=0;
end

%iProc=ij.process.ImageProcessor();
stack=ij.ImageStack(size(img,2),size(img,1));
if color==0
    for i=1:size(img,3)
        %slice=iProc.setFloatArray(img(:,:,i));
        slice=ij.process.FloatProcessor(img(:,:,i)');
        stack.addSlice('metadata',slice);
    end
else
    for i=1:size(img,4)
        sliceR=ij.process.FloatProcessor(img(:,:,1,i)');
        sliceR=ij.process.ByteProcessor(sliceR,false);
        sliceG=ij.process.FloatProcessor(img(:,:,2,i)');
        sliceG=ij.process.ByteProcessor(sliceG,false);
        sliceB=ij.process.FloatProcessor(img(:,:,3,i)');
        sliceB=ij.process.ByteProcessor(sliceB,false);

        slice=ij.process.ColorProcessor(size(img,2),size(img,1));
        slice.setChannel(1,sliceR);
        slice.setChannel(2,sliceG);
        slice.setChannel(3,sliceB);

        stack.addSlice('metadata',slice);
    end
end

iPlus=ij.ImagePlus();
iPlus.setStack(stack);

f=ij.io.FileSaver(iPlus);

f.saveAsTiff(filename);

