
% pdco_l1tf: L1 Trend Filter using PDCO to solve the dual form.
% 
% The L1 trend filter problem is
%
%       min 1/2 || y - beta ||_2^2 + lam * || D beta ||_1
%
% for some appropriate differencing matrix D and the dual problem is
%
%       min 1/2 || D' x ||_2^2 - (Dy)' x
%       wrt - lam <= x <= lam
%
% with coefficients determined by beta = y - D' x.
%
% PDCO solves problems of the form
%
%       min phi(x) + 1/2|| D1 x ||_2^2 + 1/2 ||r||_2^2
%       wrt l <= x <= u
%       sto A x + D2 r = b
%
% Set phi(x) = -(Dy)'x, D1 = eps (small), D2 = 1, A = -D', and b = 0; then
% PDCO solves
%
%       min -(Dy)'x + eps/2 ||x||_2^2 + 1/2 ||r||_2^2
%       wrt -lam <= x <= lam
%       sto -D'x + r = 0 (i.e., r = D'x)
%
% which is equivalent to the L1 Trend Filter problem up to the small
% perturbation on x in the objective. We can also change this to be
%
%       min -lam (Dy)'(x/lam) + lam^2 eps/2 ||(x/lam)||_2^2 
%                                           + lam^2/2 ||r/lam||_2^2
%       wrt -1 <= (x/lam) <= 1
%       sto - D'(x/lam) + (r/lam) = 0
% 
% which if we substitute u = x/lam, s = r/lam is
%
%       min -(Dy/lam)'u + eps/2 ||u||_2^2 + 1/2 ||s||_2^2
%       wrt -1 <= u <= 1
%       sto - D'u + s = 0
%
% This version might have better numerical properties, in that the variable
% u is guaranteed to have good scaling and known "typical" size regardless
% of penalty parameter. So here we would set phi(x) = -(Dy/lam)'x, 
% D1 = eps (small), D2 = 1, A = -D', and b = 0 with unit bounds. After a
% solution u = x/lam is obtained, we have coefficients
% 
%       beta = y - D' x = y - lam D' u
% 

% Examples:
%
%   (1) Given N (even), a > 0, real b >> a, set 
% 
%           y = a * randn(N,1), y(N/2:N) = y(N/2:N) + b
% 
%       Should be a step function, TF fit by order 0. 
%

% Author(s)
% 
%  W. Ross Morrow (WRM; mailto:morrowwr@gmail.com)
% 

% License
% 
% Covered under the GNU software license, v3.0 (https://www.gnu.org/licenses/gpl-3.0.en.html)
% 

% Notes:
%
% 28 03 2016: (WRM) First implementation. Only incorporating the "constant"
%             trend filter of zeroth order. Can be changed by inserting a
%             different differencing matrix as callable pdMat or explicit
%             sparse matrix using repeated differencing
%
% 28 03 2016: (WRM) Incorporated input from Michael Saunders, SOL lab and
%             MS&E/ICME at Stanford, primary author of PDCO software. Fixed
%             a bug in notation affecting solutions. 
% 
% 29 03 2016: (WRM) Incorporated generic order Trend Filter via
%             differencing matrix calls, although PDCO runs with an
%             explicit (sparse) matrix
% 
% 31 03 2016: (WRM) Incorporated normalization, following Michael Saunders
%             comments on the L1 outlier detection problem. This may or may
%             not be a good idea, as simple "step" function analysis shows
%             (cf. Example 1)
% 
% 02 04 2016: (WRM) Incorporated "x" coordinates for the data, as in
% 
%                 D(k+1) = D(1) diag( k ./ (x(k+1:N)-x(1:N-k)) ) D(k)
%

function [ beta , u ] = pdco_l1tf( y , x , lam , k , u0 )

    N = size(y,1);   fprintf( 'Problem size: %i\n', N )

    % Construct difference operator D   N-1 x N with 2*(N-1) nonzeros
    rows = zeros(2*(N-1),1); % Preallocate
    cols = rows;
    data = rows;
    for n = 1:N-1
        rows(2*n-1) =  n; rows(2*n) = n;
        cols(2*n-1) =  n; cols(2*n) = n+1;
        data(2*n-1) = -1; data(2*n) = 1;
    end
    D = sparse( rows, cols, data );
    clear rows cols data

    mA = N;
    nA = N-k-1;
    A = D; 
    % for i = 1:k, A <- D(i+1) = D(1) S(i) D(i) = D(1) A
    if( isempty(x) ), for i = 1:k, A = D1v( A ); end;
    else, 
        for i = 1:k, 
            A = diag(i./(x(i+1:N)-x(1:N-i))) * A; 
            A = D1v( A ); 
        end; 
    end
    A = - A'; % A = - D'
    b  =   zeros(mA,1);
    c  =   A' * y; % / lam; % c = - D y/lam
    l  = - lam * ones(nA,1);
    u  =   lam * ones(nA,1);

    opt = pdcoSet();
    opt.Method = 2;  % 22;
    opt.Print  = 1;
    opt.Wait   = 0;

    % "smarter" initial condition?
    %     rows = [ [1:N-1] , [2:N-1] , [1:N-2] ]';
    %     cols = [ [1:N-1] , [1:N-2] , [2:N-1] ]';
    %     data = [ 2*ones(N-1,1) ; -1*ones(2*(N-2),1) ];
    %     DDT = sparse( rows , cols , data , N-1 , N-1 , N-1 + 2*(N-2) );
    %     clear rows cols data,
    %     L = chol( DDT , 'lower' );
    %     x0 = - L \ nDy;
    %     x0 = L' \ x0;
    %     y0 = zeros( N   , 1 ); % dual variables on linear equalities
    %     z0 = zeros( N-1 , 1 ); % dual variables on bounds

    % if( isempty( u0 ) ), u0 = zeros( nA, 1 ); end % primal variables (for dual problem)
    u0 = zeros( nA, 1 ); % primal variables (for dual problem)
    v0 = zeros( mA, 1 ); % dual variables on linear equalities
    w0 = zeros( nA, 1 ); % dual variables on bounds
    d1 = 1e-6;
    d2 = 1;
    xsize = 1;
    zsize = 1;

    [u,v,w,inform,PDitns,CGitns,time] ...
        = pdco( c, A, b, l, u, d1, d2, opt, u0, v0, w0, xsize, zsize );
    
    beta = y + A * u;
    % beta = y + lam * A * u;

end

function Ax = multBynDk( mode , M , N , x , k )
    Ax = - x;
    if mode == 1, for i = 1:k+1, Ax = D1Tv( Ax ); end
    else, for i = 1:k+1, Ax = D1v( Ax ); end
    end
end

function r = D1v( v )
    n = size( v , 1 );
    r = v(2:n,:) - v(1:n-1,:);
end

function r = D1Tv( v )
    n = size( v , 1 );
    r = zeros( n+1 , size(v,2) );
    r( 1 ,:) = - v(1,:);
    r(2:n,:) = - v(2:n,:) + v(1:n-1,:);
    r(n+1,:) =   v(n,:);
end