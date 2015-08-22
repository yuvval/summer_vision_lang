%nel command window settare i=:
clear
clc
close all
i=0;
%for i=42:47
    switch i      
        case 0
            seq='me';
            alg='CLG-TV';
            name=strcat(seq,'_',alg);
            cd videos
            obj = VideoReader('outfile.avi');
            video = obj.read();
            cd ./..
            for k=1:3:90
            	acquisitionSequence
                main2
            end
                        
    end
    clear
    close all
    
%end