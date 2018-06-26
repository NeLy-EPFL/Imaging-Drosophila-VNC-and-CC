function f_out = crop_fit_size_center(f, target_size)

sf = size(f);
shift = round((sf - target_size)/2);
f_out = f(shift(1)+1:end-shift(1), shift(2)+1:end-shift(2), shift(3)+1:end-shift(3));

end

