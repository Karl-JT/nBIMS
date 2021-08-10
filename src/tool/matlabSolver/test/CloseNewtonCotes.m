
function w = CloseNewtonCotes(n)

  w = 0:n;
  m = fliplr(vander(w));
  for ii = 0: n
    b(ii + 1) = n^(ii + 1)/(ii + 1);
  end
  w = m'\b';
  
  return
end