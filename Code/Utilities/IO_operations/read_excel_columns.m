function data = read_excel_columns(filename,sheet,columns,firstrow,lastrow)
% Read selected columns from large Excel sheet using ActiveXServer
%  filename:  Seems that you have to use the full path to the Excel file
%  sheet   :  e.g. 'Sheet1'
%  columns :  array of column mumbers, e.g [17,341,784]
%  firstrow, lastrow:  The first and last rows to be read
%  data:   : array of numerical values
%
% Are Mjaavatten, 2016-03-14
    nrows = lastrow-firstrow+1;
    ncols = length(columns);
    data = zeros(nrows,ncols);
    first = num2str(firstrow);
    last = num2str(lastrow);
    hExcel = actxserver('Excel.Application');
    hWorkbook = hExcel.Workbooks.Open(filename);
    hWorksheet = hWorkbook.Sheets.Item(sheet);
    failed = 0;
    T  = table();
    try
        for i = 1:ncols
            col = col2str(columns(i));
            Range = [col,first,':',col,last];
            RangeObj = hWorksheet.Range(Range);
            values = RangeObj.value;
            is_char = ischar(values{2});
            is_num = isnumeric(values{2});
            name = values{1};
            if is_num 
                to_table = cell2mat(values(2:end));
            else
               to_table = values(2:end);
            end
                
            t = table(to_table, 'VariableNames', {name});
            T = [T,t];
           
        end

    catch
        failed = 1;
        release(hWorksheet)
        release(hWorkbook)
        release(hExcel)
    end
    if ~failed
        release(hWorksheet)
        release(hWorkbook)
        release(hExcel)
    end
    end
    function colname = col2str(n)
    % Translate Excel column number to Column characters
    s = '';
    while n > 0
        s = [s,char(mod(n-1,26)+65)];
        n = floor((n-1)/26);
    end
    colname = deblank(fliplr(s));
    end