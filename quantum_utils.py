import numpy as np
import cmd
import sys
import re
from fractions import Fraction

class QuantumUtils(cmd.Cmd):
    intro = """
╔══════════════════════════════════════════════════════════════════╗
║                    QuantumUtils v1.0.0                           ║
║     The easy-to-use tool for Quantum Computing Calculations      ║
║                   Type "help" for commands                       ║
╚══════════════════════════════════════════════════════════════════╝
"""
    prompt = 'quantum> '

    def __init__(self):
        super().__init__()
        self.variables = {}
        self.pauli_matrices = {
            'I': np.array([[1, 0], [0, 1]], dtype=complex),
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex),
            'H': np.array([[1/np.sqrt(2), 1/np.sqrt(2)],
                          [1/np.sqrt(2), -1/np.sqrt(2)]], dtype=complex)
        }

    def preloop(self):
        """Initialize with common quantum states and operators"""
        # Common quantum states
        self.variables['|0>'] = np.array([[1], [0]], dtype=complex)
        self.variables['|1>'] = np.array([[0], [1]], dtype=complex)
        self.variables['|+>'] = np.array([[1/np.sqrt(2)], [1/np.sqrt(2)]], dtype=complex)
        self.variables['|->'] = np.array([[1/np.sqrt(2)], [-1/np.sqrt(2)]], dtype=complex)

        # Add Pauli matrices to variables
        for name, matrix in self.pauli_matrices.items():
            self.variables[name] = matrix

    def do_quit(self, arg):
        """Exit QuantumUtils"""
        print("Thank you for using QuantumUtils!")
        return True

    def do_exit(self, arg):
        """Exit QuantumUtils"""
        return self.do_quit(arg)

    def do_help(self, arg):
        """Show help menu"""
        help_text = """
General Commands:
  help, exit, clear, list

Matrix Operations:
  create <name> <rows> <cols> - Create a new matrix
  show <name> - Display a matrix/operator in matrix form
  add <A> <B> <result> - Matrix addition: result = A + B
  mult <A> <B> <result> - Matrix multiplication: result = A * B
  tensor <A> <B> <result> - Tensor product: result = A \otimes B
  det <A> - Calculate determinant of matrix A
  eigen <A> - Calculate eigenvalues and eigenvectors of matrix A
  transpose <A> <result> - Transpose of matrix A
  dagger <A> <result> - Conjugate transpose (dagger) of matrix A
  inverse <A> <result> - Matrix inverse of A

Quantum Operations:
  inner <bra| |ket> - Calculate inner product <bra|ket>
  outer <ket> <bra> - Calculate outer product |ket><bra|
  normalize <state> <result> - Normalize a state vector
  expectation <operator> <state> - Calculate expectation value of an operator

Special Functions:
  pauli <name> - Show Pauli matrix (I, X, Y, Z, H)
  bell - Create Bell states

Examples:
  create my_matrix 2 2
  add X Y result
  tensor |0> |1> result
  inner <0| |1>
  expectation X |+>
"""
        print(help_text)

    def do_clear(self, arg):
        """Clear the terminal"""
        print("\033[H\033[J") # \033[H\033[J sequence represents a pair of ANSI escape codes used to control a terminal display.

    def do_list(self, arg):
        """List all variables"""
        print("Stored variables:")
        for name, value in self.variables.items():
            if isinstance(value, np.ndarray):
                print(f"  {name}: {value.shape} array")
            else:
                print(f"  {name}: {type(value).__name__}")

    def do_create(self, arg):
        """Create a new matrix: create <name> <rows> <cols>"""
        args = arg.split()
        if len(args) != 3:
            print("Usage: create <name> <rows> <cols>")
            return

        name, rows_str, cols_str = args
        try:
            rows, cols = int(rows_str), int(cols_str)
            print(f"Enter {rows}x{cols} matrix (complex numbers: a+bj):")
            matrix = []
            for i in range(rows):
                row = input(f"Row {i+1}: ").split()
                if len(row) != cols:
                    print(f"Error: Expected {cols} elements per row")
                    return
                complex_row = []
                for elem in row:
                    try:
                        # Handle complex number input
                        if 'j' in elem:
                            # Replace i with j if needed
                            elem = elem.replace('i', 'j')
                        complex_row.append(complex(elem))
                    except ValueError:
                        print(f"Error: Invalid complex number: {elem}")
                        return
                matrix.append(complex_row)

            self.variables[name] = np.array(matrix, dtype=complex)
            print(f"Matrix {name} created successfully!")

        except ValueError:
            print("Error: Rows and columns must be integers")

    def do_show(self, arg):
        """Display a matrix/variable: show <name>"""
        if not arg:
            print("Usage: show <name>")
            return

        if arg not in self.variables:
            print(f"Error: Variable '{arg}' not found")
            return

        value = self.variables[arg]
        if isinstance(value, np.ndarray):
            print(f"{arg} =")
            self._print_matrix(value)
        else:
            print(f"{arg} = {value}")

    def do_add(self, arg):
        """Matrix addition: add <A> <B> <result>"""
        args = arg.split()
        if len(args) != 3:
            print("Usage: add <A> <B> <result>")
            return

        A_name, B_name, result_name = args
        if A_name not in self.variables or B_name not in self.variables:
            print("Error: One or both matrices not found")
            return

        A = self.variables[A_name]
        B = self.variables[B_name]

        if A.shape != B.shape:
            print("Error: Matrices must have the same dimensions")
            return

        try:
            result = A + B
            self.variables[result_name] = result
            print(f"Result stored in {result_name}:")
            self._print_matrix(result)
        except Exception as e:
            print(f"Error: {e}")

    def do_mult(self, arg):
        """Matrix multiplication: mult <A> <B> <result>"""
        args = arg.split()
        if len(args) != 3:
            print("Usage: mult <A> <B> <result>")
            return

        A_name, B_name, result_name = args
        if A_name not in self.variables or B_name not in self.variables:
            print("Error: One or both matrices not found")
            return

        A = self.variables[A_name]
        B = self.variables[B_name]

        if A.shape[1] != B.shape[0]:
            print("Error: Matrix dimensions not compatible for multiplication")
            return

        try:
            result = np.matmul(A, B)
            self.variables[result_name] = result
            print(f"Result stored in {result_name}:")
            self._print_matrix(result)
        except Exception as e:
            print(f"Error: {e}")

    def do_tensor(self, arg):
        """Tensor product: tensor <A> <B> <result>"""
        args = arg.split()
        if len(args) != 3:
            print("Usage: tensor <A> <B> <result>")
            return

        A_name, B_name, result_name = args
        if A_name not in self.variables or B_name not in self.variables:
            print("Error: One or both matrices not found")
            return

        A = self.variables[A_name]
        B = self.variables[B_name]

        try:
            result = np.kron(A, B)
            self.variables[result_name] = result
            print(f"Result stored in {result_name}:")
            self._print_matrix(result)
        except Exception as e:
            print(f"Error: {e}")

    def do_det(self, arg):
        """Calculate determinant of a matrix: det <A>"""
        if not arg:
            print("Usage: det <A>")
            return

        if arg not in self.variables:
            print(f"Error: Variable '{arg}' not found")
            return

        A = self.variables[arg]
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            print("Error: Matrix must be square")
            return

        try:
            det = np.linalg.det(A)
            print(f"Det({arg}) = {det}")
        except Exception as e:
            print(f"Error: {e}")

    def do_eigen(self, arg):
        """Calculate eigenvalues and eigenvectors: eigen <A>"""
        if not arg:
            print("Usage: eigen <A>")
            return

        if arg not in self.variables:
            print(f"Error: Variable '{arg}' not found")
            return

        A = self.variables[arg]
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            print("Error: Matrix must be square")
            return

        try:
            eigenvalues, eigenvectors = np.linalg.eig(A)
            print(f"Eigenvalues of {arg}:")
            for i, val in enumerate(eigenvalues):
                print(f"  λ_{i} = {val}")

            print(f"\nEigenvectors of {arg}:")
            for i in range(eigenvectors.shape[1]):
                print(f"  v_{i} = {eigenvectors[:, i]}")
        except Exception as e:
            print(f"Error: {e}")

    def do_conjugate(self, arg):
        """Complex conjugate: conjugate <A> <result>"""
        args = arg.split()
        if len(args) != 2:
            print("Usage: conjugate <A> <result>")
            return

        A_name, result_name = args
        if A_name not in self.variables:
            print(f"Error: Variable '{A_name}' not found")
            return

        A = self.variables[A_name]

        try:
            result = np.conj(A)
            self.variables[result_name] = result
            print(f"Result stored in {result_name}:")
            self._print_matrix(result)
        except Exception as e:
            print(f"Error: {e}")

    def do_transpose(self, arg):
        """Matrix transpose: transpose <A> <result>"""
        args = arg.split()
        if len(args) != 2:
            print("Usage: transpose <A> <result>")
            return

        A_name, result_name = args
        if A_name not in self.variables:
            print(f"Error: Variable '{A_name}' not found")
            return

        A = self.variables[A_name]

        try:
            result = np.transpose(A)
            self.variables[result_name] = result
            print(f"Result stored in {result_name}:")
            self._print_matrix(result)
        except Exception as e:
            print(f"Error: {e}")

    def do_dagger(self, arg):
        """Conjugate transpose (dagger): dagger <A> <result>"""
        args = arg.split()
        if len(args) != 2:
            print("Usage: dagger <A> <result>")
            return

        A_name, result_name = args
        if A_name not in self.variables:
            print(f"Error: Variable '{A_name}' not found")
            return

        A = self.variables[A_name]

        try:
            result = np.conj(np.transpose(A))
            self.variables[result_name] = result
            print(f"Result stored in {result_name}:")
            self._print_matrix(result)
        except Exception as e:
            print(f"Error: {e}")

    def do_inverse(self, arg):
        """Matrix inverse: inverse <A> <result>"""
        args = arg.split()
        if len(args) != 2:
            print("Usage: inverse <A> <result>")
            return

        A_name, result_name = args
        if A_name not in self.variables:
            print(f"Error: Variable '{A_name}' not found")
            return

        A = self.variables[A_name]
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            print("Error: Matrix must be square")
            return

        try:
            result = np.linalg.inv(A)
            self.variables[result_name] = result
            print(f"Result stored in {result_name}:")
            self._print_matrix(result)
        except np.linalg.LinAlgError:
            print("Error: Matrix is singular and cannot be inverted")
        except Exception as e:
            print(f"Error: {e}")

    def do_inner(self, arg):
        """Inner product: inner <bra| |ket>"""
        # Parse input with special handling for bra-ket notation
        match = re.match(r'<(.*)\| \|?(.*)>', arg)
        if not match:
            print("Usage: inner <bra| |ket>")
            return

        bra_name, ket_name = match.groups()
        bra_name = bra_name.strip()
        ket_name = ket_name.strip()

        # Handle bra notation - convert to ket and then take dagger
        if bra_name not in self.variables:
            # Try to find the corresponding ket
            ket_for_bra = f"|{bra_name}>"
            if ket_for_bra not in self.variables:
                print(f"Error: Variable '{bra_name}' not found")
                return
            bra = np.conj(self.variables[ket_for_bra].T)
        else:
            bra = self.variables[bra_name]

        # Handle ket notation
        ket_var_name = f"|{ket_name}>"
        if ket_var_name not in self.variables:
            print(f"Error: Variable '{ket_name}' not found")
            return
        ket = self.variables[ket_var_name]

        # Check dimensions
        if bra.shape[1] != ket.shape[0]:
            print("Error: Vectors have incompatible dimensions")
            return

        try:
            result = np.matmul(bra, ket)
            print(f"<{bra_name}|{ket_name}> = {result[0, 0]}")
        except Exception as e:
            print(f"Error: {e}")

    def do_outer(self, arg):
      """Outer product: outer |ket> <bra|"""
      #Parse input with special handling for bra-ket notation
      match = re.match(r'\|(.*)> <(.*)\|', arg)
      if not match:
          print("Usage: outer |ket> <bra|")
          return

      ket_name, bra_name = match.groups()
      ket_name = ket_name.strip()
      bra_name = bra_name.strip()

      # Handle ket notation
      ket_var_name = f"|{ket_name}>"
      if ket_var_name not in self.variables:
          print(f"Error: Variable '{ket_var_name}' not found")
          return
      ket = self.variables[ket_var_name]

      # Handle bra notation. change to ket and then take dagger
      bra_ket_var_name = f"|{bra_name}>"
      if bra_ket_var_name not in self.variables:
          print(f"Error: Variable '{bra_ket_var_name}' not found")
          return
      bra = np.conj(self.variables[bra_ket_var_name].T)

      # Check dimensions
      if ket.shape[1] != 1:
          print("Error: Ket must be a column vector")
          return

      try:
          result = np.outer(ket, bra)
          result_name = f"|{ket_name}><{bra_name}|"
          self.variables[result_name] = result
          print(f"Result stored in {result_name}:")
          self._print_matrix(result)
      except Exception as e:
          print(f"Error: {e}")

    def do_normalize(self, arg):
        """Normalize a state vector: normalize <state> <result>"""
        args = arg.split()
        if len(args) != 2:
            print("Usage: normalize <state> <result>")
            return

        state_name, result_name = args
        if state_name not in self.variables:
            print(f"Error: Variable '{state_name}' not found")
            return

        state = self.variables[state_name]
        if state.ndim != 2 or state.shape[1] != 1:
            print("Error: Input must be a column vector")
            return

        try:
            norm = np.linalg.norm(state)
            if norm == 0:
                print("Error: Cannot normalize zero vector")
                return

            result = state / norm
            self.variables[result_name] = result
            print(f"Normalized state stored in {result_name}:")
            self._print_matrix(result)
        except Exception as e:
            print(f"Error: {e}")

    def do_expectation(self, arg):
        """Expectation value: expectation <operator> <state>"""
        args = arg.split()
        if len(args) != 2:
            print("Usage: expectation <operator> <state>")
            return

        op_name, state_name = args
        if op_name not in self.variables:
            print(f"Error: Variable '{op_name}' not found")
            return

        if state_name not in self.variables:
            print(f"Error: Variable '{state_name}' not found")
            return

        op = self.variables[op_name]
        state = self.variables[state_name]

        if state.ndim != 2 or state.shape[1] != 1:
            print("Error: State must be a column vector")
            return

        if op.ndim != 2 or op.shape[0] != op.shape[1]:
            print("Error: Operator must be a square matrix")
            return

        if op.shape[0] != state.shape[0]:
            print("Error: Operator and state have incompatible dimensions")
            return

        try:
            bra = np.conj(state.T)
            intermediate = np.matmul(op, state)
            expectation = np.matmul(bra, intermediate)
            print(f"<{state_name}|{op_name}|{state_name}> = {expectation[0, 0]}")
        except Exception as e:
            print(f"Error: {e}")

    def do_pauli(self, arg):
        """Show Pauli matrix: pauli <name> (I, X, Y, Z, H)"""
        if not arg:
            print("Usage: pauli <name> (I, X, Y, Z, H)")
            return

        if arg not in self.pauli_matrices:
            print(f"Error: Pauli matrix '{arg}' not found. Available: I, X, Y, Z, H")
            return

        matrix = self.pauli_matrices[arg]
        print(f"{arg} =")
        self._print_matrix(matrix)

    def do_bell(self, arg):
        """Create Bell states: bell"""
        # Create the four Bell states
        bell00 = np.kron(self.variables['|0>'], self.variables['|0>']) + \
                 np.kron(self.variables['|1>'], self.variables['|1>'])
        bell00 /= np.sqrt(2)

        bell01 = np.kron(self.variables['|0>'], self.variables['|0>']) - \
                 np.kron(self.variables['|1>'], self.variables['|1>'])
        bell01 /= np.sqrt(2)

        bell10 = np.kron(self.variables['|0>'], self.variables['|1>']) + \
                 np.kron(self.variables['|1>'], self.variables['|0>'])
        bell10 /= np.sqrt(2)

        bell11 = np.kron(self.variables['|0>'], self.variables['|1>']) - \
                 np.kron(self.variables['|1>'], self.variables['|0>'])
        bell11 /= np.sqrt(2)

        self.variables['|PHI+>'] = bell00
        self.variables['|PHI->'] = bell01
        self.variables['|PSI+>'] = bell10
        self.variables['|PSI->'] = bell11

        print("Bell states created:")
        print("  |PHI+> = (|00> + |11>)/sqrt{2}")
        print("  |PHI-> = (|00> - |11>)/sqrt{2}")
        print("  |PSI+> = (|01> + |10>)/sqrt{2}")
        print("  |PSI-> = (|01> - |10>)/sqrt{2}")

    def _print_matrix(self, matrix):
        """Helper function to print matrices in a readable format"""
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)

        rows, cols = matrix.shape

        for i in range(rows):
            row_str = "  "
            for j in range(cols):
                elem = matrix[i, j]
                #Formats complex numbers nicely
                if elem.imag == 0:
                    row_str += f"{elem.real:8.3f}  "
                elif elem.real == 0:
                    row_str += f"{elem.imag:8.3f}j  "
                else:
                    sign = '+' if elem.imag >= 0 else '-'
                    row_str += f"{elem.real:6.3f} {sign} {abs(elem.imag):.3f}j  "
            print(row_str)

    def default(self, line):
        """Handle special bra-ket notation input"""
        # Check if input is in bra-ket notation
        ket_match = re.match(r'\|(.*)>', line)
        bra_match = re.match(r'<(.*)\|', line)

        if ket_match:
            # It's a ket check if we're creating or showing
            ket_name = ket_match.group(1)
            if ket_name in self.variables:
                self.do_show(f"|{ket_name}>")
            else:
                print(f"Error: Ket |{ket_name}> not found")
        elif bra_match:
            # It's a bra, check if we're creating or showing
            bra_name = bra_match.group(1)
            if bra_name in self.variables:
                # Convert ket to bra by taking conjugate transpose / dagger
                ket = self.variables[bra_name]
                bra = np.conj(ket.T)
                bra_name_display = f"<{bra_name}|"
                self.variables[bra_name_display] = bra
                print(f"Bra {bra_name_display} created from ket |{bra_name}>")
            else:
                print(f"Error: Ket |{bra_name}> not found to create bra <{bra_name}|")
        else:
            print(f"Error: Unknown command '{line}'")

if __name__ == '__main__':
    try:
        QuantumUtils().cmdloop()
    except KeyboardInterrupt:
        print("\n\nThank you for using QuantumUtils!")
        sys.exit(0)