function Login() {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gradient-to-r from-blue-500 to-teal-500">
        <div className="w-full max-w-md p-8 space-y-6 bg-white rounded-lg shadow-2xl animate-fade-in-down">
          <h2 className="text-3xl font-extrabold text-center text-gray-800">
            Welcome Back
          </h2>
          <p className="text-sm text-center text-gray-500">
            Please sign in to your account.
          </p>
          <form className="mt-8 space-y-6">
            <div className="space-y-4">
              <div>
                <label htmlFor="email" className="block text-sm font-medium text-gray-700">
                  Email Address
                </label>
                <input
                  type="email"
                  id="email"
                  name="email"
                  required
                  className="block w-full px-4 py-3 mt-1 text-gray-900 placeholder-gray-400 bg-gray-50 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-teal-500 focus:border-teal-500"
                  placeholder="you@example.com"
                />
              </div>
              <div>
                <label htmlFor="password" className="block text-sm font-medium text-gray-700">
                  Password
                </label>
                <input
                  type="password"
                  id="password"
                  name="password"
                  required
                  className="block w-full px-4 py-3 mt-1 text-gray-900 placeholder-gray-400 bg-gray-50 border border-gray-300 rounded-lg shadow-sm focus:outline-none focus:ring-teal-500 focus:border-teal-500"
                  placeholder="Enter your password"
                />
              </div>
            </div>
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="remember-me"
                  name="remember-me"
                  className="w-4 h-4 text-teal-600 border-gray-300 rounded focus:ring-teal-500"
                />
                <label htmlFor="remember-me" className="ml-2 text-sm text-gray-600">
                  Remember me
                </label>
              </div>
              <a href="#" className="text-sm font-medium text-teal-600 hover:underline">
                Forgot password?
              </a>
            </div>
            <button
              type="submit"
              className="w-full px-4 py-3 text-sm font-bold text-white bg-teal-600 rounded-lg shadow-lg hover:bg-teal-500 focus:outline-none focus:ring-4 focus:ring-teal-300"
            >
              Sign In
            </button>
          </form>
          <p className="text-sm text-center text-gray-600">
            Don’t have an account?{' '}
            <a href="#" className="font-medium text-teal-600 hover:underline">
              Sign Up
            </a>
          </p>
        </div>
      </div>
    );
  }

export default Login
  