using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PokemonPSAv3
{
    class WordGroup
    {
        public List<string> Words { get; private set; }

        public int X1 { get; private set; }

        public int X2 { get; private set; }

        public WordGroup(List<string> words, int x1, int x2)
        {
            Words = words;
            X1 = x1;
            X2 = x2;
        }

        public override string ToString()
        {
            return $"WordGroup({X1}, {X2}, [{string.Join(", ", Words)}])";
        }
    }
}
